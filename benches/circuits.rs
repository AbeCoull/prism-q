//! Macrobenchmarks: circuit family sweeps (qubit count and depth).
//!
//! Use `--features bench-fast` for a quick run that reduces warmup and
//! measurement time. Omit for the full suite with default Criterion timing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_q::backend::Backend;
use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::{BackendKind, MpsBackend};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

const SEED: u64 = 0xDEAD_BEEF;

fn is_fast() -> bool {
    cfg!(feature = "bench-fast")
}

fn configure_group(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    if is_fast() {
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
    } else {
        group.sample_size(10);
    }
}

// ---- Bench-specific circuit builders (not shared) ----

fn qft_like_circuit(n_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for i in 0..n_qubits {
        circuit.add_gate(Gate::H, &[i]);
        for j in (i + 1)..n_qubits {
            let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
            circuit.add_gate(Gate::cphase(theta), &[i, j]);
        }
    }

    circuit
}

fn sparse_entanglement_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for _ in 0..depth {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
        }
        if n_qubits >= 2 {
            circuit.add_gate(Gate::Cx, &[0, n_qubits - 1]);
        }
    }

    circuit
}

fn dense_entanglement_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for _ in 0..depth {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
        }
        for q in 0..n_qubits - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    circuit
}

fn random_clifford_circuit(n_qubits: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];

    for layer in 0..depth {
        for q in 0..n_qubits {
            let gate_idx = rng.random_range(0..cliffords.len());
            circuit.add_gate(cliffords[gate_idx].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n_qubits - 1).step_by(2) {
            if rng.random_bool(0.5) {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    circuit
}

fn random_single_qubit_circuit(n_qubits: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let gates = [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::T];

    for _ in 0..depth {
        for q in 0..n_qubits {
            let gate_idx = rng.random_range(0..gates.len());
            circuit.add_gate(gates[gate_idx].clone(), &[q]);
        }
    }
    circuit
}

fn mps_adjacent_phase_circuit(n_qubits: usize, rounds: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }

    for _ in 0..rounds {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::Ry(0.17 + q as f64 * 0.003), &[q]);
        }
        for q in (0..n_qubits.saturating_sub(1)).step_by(2) {
            circuit.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_8), &[q, q + 1]);
        }
    }

    circuit
}

fn mps_long_range_phase_circuit(n_qubits: usize, rounds: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }

    for _ in 0..rounds {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::Ry(0.17 + q as f64 * 0.003), &[q]);
        }
        for q in 0..(n_qubits / 2) {
            let partner = n_qubits - 1 - q;
            circuit.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_8), &[q, partner]);
        }
    }

    circuit
}

fn mps_measure_reset_circuit(n_qubits: usize, rounds: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, n_qubits);

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n_qubits.saturating_sub(1) {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }

    for round in 0..rounds {
        let parity = round % 2;
        for q in (parity..n_qubits).step_by(2) {
            circuit.add_measure(q, q);
            circuit.add_reset(q);
            circuit.add_gate(Gate::H, &[q]);
            if q + 1 < n_qubits {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }

    circuit
}

fn run_mps_apply_only(circuit: &Circuit, max_bond_dim: usize) {
    let mut backend = MpsBackend::new(SEED, max_bond_dim);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    black_box(backend.classical_results());
}

// ---- Statevector: qubit-count sweeps ----

fn bench_statevector_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/random_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qft_like");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16] {
        let circuit = qft_like_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_hea(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/hea_l5");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_clifford(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/clifford_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_qft_textbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qft_textbook");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20, 22] {
        let circuit = circuits::qft_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_qpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qpe_t_gate");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20, 22] {
        let circuit = circuits::phase_estimation_circuit(n);
        let label = format!("{}q", n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Statevector: QAOA ----

fn bench_statevector_qaoa(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qaoa_l3");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::qaoa_circuit(n, 3, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Statevector: depth sweep ----

fn bench_statevector_depth_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/depth_sweep_12q");
    configure_group(&mut group);

    let depths = vec![5, 10, 20, 50, 100];
    for &depth in &depths {
        let circuit = circuits::random_circuit(12, depth, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(depth), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Statevector: entanglement structure ----

fn bench_statevector_entanglement(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/entanglement_16q_d10");
    configure_group(&mut group);

    let sparse = sparse_entanglement_circuit(16, 10);
    group.bench_function("sparse", |b| {
        b.iter(|| {
            sim::run_with(BackendKind::Statevector, &sparse, 42).unwrap();
        });
    });

    let dense = dense_entanglement_circuit(16, 10);
    group.bench_function("dense", |b| {
        b.iter(|| {
            sim::run_with(BackendKind::Statevector, &dense, 42).unwrap();
        });
    });

    group.finish();
}

// ---- Statevector: scalability sweep (2–26 qubits) ----

fn bench_statevector_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/scalability_d5");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    for n in (2..=26).step_by(2) {
        let circuit = circuits::random_circuit(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Stabilizer backend ----

fn bench_stabilizer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer/scaling");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500, 1000] {
        let circuit = random_clifford_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Stabilizer, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_stabilizer_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer/measurement");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500, 1000] {
        let mut circuit = Circuit::new(n, n);
        circuit.add_gate(Gate::H, &[0]);
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        group.bench_with_input(
            BenchmarkId::new("ghz_measure_all", n),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    sim::run_with(BackendKind::Stabilizer, circ, 42).unwrap();
                });
            },
        );
    }
    group.finish();
}

// ---- Factored stabilizer backend ----

fn bench_factored_stabilizer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored_stabilizer/scaling");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500, 1000] {
        let circuit = random_clifford_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_factored_stabilizer_local(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored_stabilizer/local_blocks");
    configure_group(&mut group);

    for &(blocks, block_size) in &[(5, 2), (5, 4), (10, 5), (20, 5), (50, 5)] {
        let n = blocks * block_size;
        let circuit = circuits::local_clifford_blocks(blocks, block_size, 10, SEED);
        group.bench_with_input(
            BenchmarkId::new(format!("{}x{}", blocks, block_size), n),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    sim::run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_factored_stabilizer_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored_stabilizer/measurement");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500] {
        let mut circuit = Circuit::new(n, n);
        circuit.add_gate(Gate::H, &[0]);
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        group.bench_with_input(
            BenchmarkId::new("ghz_measure_all", n),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    sim::run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
                });
            },
        );
    }
    group.finish();
}

// ---- Sparse backend ----

fn bench_sparse_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse/random_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Sparse, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_sparse_low_entanglement(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse/low_entanglement");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = sparse_entanglement_circuit(n, 5);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Sparse, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

// ---- MPS backend ----

fn bench_mps_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps/random_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Mps { max_bond_dim: 64 }, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_mps_linear_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps/linear_chain_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 16, 32, 64] {
        let circuit = dense_entanglement_circuit(n, 10);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Mps { max_bond_dim: 64 }, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_mps_hotspots(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps/hotspots");
    configure_group(&mut group);

    for &n in &[16, 32] {
        let adjacent = mps_adjacent_phase_circuit(n, 4);
        group.bench_with_input(
            BenchmarkId::new("adjacent_cp_r4", n),
            &adjacent,
            |b, circ| {
                b.iter(|| run_mps_apply_only(circ, 64));
            },
        );

        let long_range = mps_long_range_phase_circuit(n, 4);
        group.bench_with_input(
            BenchmarkId::new("long_range_cp_r4", n),
            &long_range,
            |b, circ| {
                b.iter(|| run_mps_apply_only(circ, 64));
            },
        );
    }

    let meas_reset = mps_measure_reset_circuit(32, 3);
    group.bench_function("measure_reset_32q_r3", |b| {
        b.iter(|| run_mps_apply_only(&meas_reset, 64));
    });

    group.finish();
}

// ---- Product state backend ----

fn bench_product_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("product/scaling_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 16, 32, 64, 128, 256] {
        let circuit = random_single_qubit_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::ProductState, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

// ---- Tensor Network backend ----

fn bench_tn_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/random_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::TensorNetwork, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_tn_linear_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/linear_chain");
    configure_group(&mut group);

    for &n in &[4, 8, 12] {
        let circuit = dense_entanglement_circuit(n, 5);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::TensorNetwork, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

// ---- Cross-backend comparisons ----

fn bench_compare_clifford(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/clifford_d10");
    configure_group(&mut group);

    let backends: &[(&str, BackendKind)] = &[
        ("statevector", BackendKind::Statevector),
        ("stabilizer", BackendKind::Stabilizer),
        ("sparse", BackendKind::Sparse),
        ("mps_64", BackendKind::Mps { max_bond_dim: 64 }),
        ("auto", BackendKind::Auto),
    ];

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = random_clifford_circuit(n, 10, SEED);
        for &(name, ref kind) in backends {
            group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                b.iter(|| {
                    sim::run_with(kind.clone(), circ, 42).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_compare_single_qubit(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/single_qubit_d10");
    configure_group(&mut group);

    let backends: &[(&str, BackendKind)] = &[
        ("statevector", BackendKind::Statevector),
        ("product", BackendKind::ProductState),
        ("auto", BackendKind::Auto),
    ];

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = random_single_qubit_circuit(n, 10, SEED);
        for &(name, ref kind) in backends {
            group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                b.iter(|| {
                    sim::run_with(kind.clone(), circ, 42).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_compare_general(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/general_d10");
    configure_group(&mut group);

    let backends: &[(&str, BackendKind)] = &[
        ("statevector", BackendKind::Statevector),
        ("sparse", BackendKind::Sparse),
        ("mps_64", BackendKind::Mps { max_bond_dim: 64 }),
        ("auto", BackendKind::Auto),
    ];

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        for &(name, ref kind) in backends {
            group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                b.iter(|| {
                    sim::run_with(kind.clone(), circ, 42).unwrap();
                });
            });
        }
    }
    group.finish();
}

// ---- Auto dispatch sweeps ----

fn bench_auto_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/random_d10");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/qft_like");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16] {
        let circuit = qft_like_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_qft_textbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/qft_textbook");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20, 22] {
        let circuit = circuits::qft_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_qpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/qpe_t_gate");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20, 22] {
        let circuit = circuits::phase_estimation_circuit(n);
        let label = format!("{}q", n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_hea(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/hea_l5");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_clifford(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/clifford_d10");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20] {
        let circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/scalability_d5");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    for n in (2..=26).step_by(2) {
        let circuit = circuits::random_circuit(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Decomposition benchmarks ----

fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition");
    configure_group(&mut group);

    for &total_q in &[8, 12, 16, 20] {
        let n_pairs = total_q / 2;
        let circuit = circuits::independent_bell_pairs(n_pairs);

        group.bench_with_input(
            BenchmarkId::new("bell_decomposed", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bell_monolithic", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    let mut sv = prism_q::StatevectorBackend::new(42);
                    sim::run_on(&mut sv, circ).unwrap();
                });
            },
        );
    }

    for &block_size in &[2, 4, 5, 10] {
        let num_blocks = 20 / block_size;
        let circuit = circuits::independent_random_blocks(num_blocks, block_size, 5, SEED);

        group.bench_with_input(
            BenchmarkId::new(format!("20q_block{}_decomposed", block_size), block_size),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("20q_block{}_monolithic", block_size), block_size),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    let mut sv = prism_q::StatevectorBackend::new(42);
                    sim::run_on(&mut sv, circ).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ---- Factored backend benchmarks ----

fn bench_factored_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/random_d10");
    configure_group(&mut group);

    for &n in &[16, 20, 24] {
        let circuit = circuits::random_circuit(n, 10, SEED);

        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| sim::run_with(BackendKind::Statevector, circ, 42).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| sim::run_with(BackendKind::Factored, circ, 42).unwrap());
        });
    }

    group.finish();
}

fn bench_factored_independent(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/independent");
    configure_group(&mut group);

    for &total_q in &[16, 20, 24] {
        let circuit = circuits::independent_random_blocks(total_q / 4, 4, 5, SEED);

        group.bench_with_input(
            BenchmarkId::new("statevector", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| sim::run_with(BackendKind::Statevector, circ, 42).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("factored", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| sim::run_with(BackendKind::Factored, circ, 42).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_factored_sim_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/sim_only_d10");
    configure_group(&mut group);

    for &n in &[16, 20, 24] {
        let circuit = circuits::random_circuit(n, 10, SEED);

        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| sim::run_with(BackendKind::Statevector, circ, 42).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| sim::run_with(BackendKind::Factored, circ, 42).unwrap());
        });
    }

    group.finish();
}

fn bench_factored_dynamic(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/dynamic_advantage");
    configure_group(&mut group);

    // Independent blocks with ONE bridging CX at the end.
    // Static analysis: single component (union-find merges all).
    // Dynamic reality: groups stay small until the bridging CX.
    for &total_q in &[16, 20, 24] {
        let block_size = 4;
        let num_blocks = total_q / block_size;
        let mut circuit = circuits::independent_random_blocks(num_blocks, block_size, 5, SEED);
        // Add one CX bridging first and last block (prevents static decomposition)
        circuit.add_gate(Gate::Cx, &[0, total_q - 1]);

        group.bench_with_input(
            BenchmarkId::new("statevector", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| sim::run_with(BackendKind::Statevector, circ, 42).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("factored", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| sim::run_with(BackendKind::Factored, circ, 42).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_factored_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/dense");
    configure_group(&mut group);

    for &n in &[12, 16, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 3, SEED);

        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| sim::run_with(BackendKind::Statevector, circ, 42).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| sim::run_with(BackendKind::Factored, circ, 42).unwrap());
        });
    }

    group.finish();
}

fn clifford_t_circuit(n_qubits: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];

    // Clifford layer first
    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n_qubits - 1 {
        if rng.random_bool(0.5) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    // Insert T gates on random qubits
    for _ in 0..t_count {
        let q = rng.random_range(0..n_qubits);
        circuit.add_gate(Gate::T, &[q]);
    }

    // More Clifford layers
    for _ in 0..3 {
        for q in 0..n_qubits {
            let gate_idx = rng.random_range(0..cliffords.len());
            circuit.add_gate(cliffords[gate_idx].clone(), &[q]);
        }
        for q in 0..n_qubits - 1 {
            if rng.random_bool(0.5) {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    circuit
}

fn bench_clifford_t(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_t");
    configure_group(&mut group);

    let n = 10;
    for &t in &[1, 2, 4, 8, 12] {
        let circuit = clifford_t_circuit(n, t, SEED);

        group.bench_function(BenchmarkId::new("spd", format!("{n}q_{t}t")), |b| {
            b.iter(|| {
                prism_q::run_spd(&circuit, 0.0, 0).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("spp_10k", format!("{n}q_{t}t")), |b| {
            b.iter(|| {
                prism_q::run_spp(&circuit, 10_000, 42).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("statevector", format!("{n}q_{t}t")), |b| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    for &n in &[5, 15, 20] {
        let circuit = clifford_t_circuit(n, 4, SEED);

        group.bench_function(BenchmarkId::new("spd", format!("{n}q_4t")), |b| {
            b.iter(|| {
                prism_q::run_spd(&circuit, 0.0, 0).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("statevector", format!("{n}q_4t")), |b| {
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_stabilizer_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_rank");
    configure_group(&mut group);

    // Compare stabilizer_rank exact vs SPD vs statevector
    let n = 10;
    for &t in &[2, 4, 8, 12] {
        let circuit = clifford_t_circuit(n, t, SEED);
        let id = format!("{n}q_{t}t");

        group.bench_function(BenchmarkId::new("stab_rank", &id), |b| {
            b.iter(|| {
                prism_q::run_stabilizer_rank(&circuit, 42).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("spd", &id), |b| {
            b.iter(|| {
                prism_q::run_spd(&circuit, 0.0, 0).unwrap();
            });
        });
    }

    // Approximate mode: higher T-counts with bounded terms
    for &t in &[16, 20] {
        let circuit = clifford_t_circuit(n, t, SEED);
        let id = format!("{n}q_{t}t");

        group.bench_function(BenchmarkId::new("approx_256", &id), |b| {
            b.iter(|| {
                prism_q::run_stabilizer_rank_approx(&circuit, 256, 42).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("approx_1024", &id), |b| {
            b.iter(|| {
                prism_q::run_stabilizer_rank_approx(&circuit, 1024, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_compiled_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_sampler");
    configure_group(&mut group);

    for &n in &[100, 500, 1000] {
        let mut circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let id = format!("noiseless_{}q_10k", n);
        group.bench_function(BenchmarkId::new("noiseless", &id), |b| {
            b.iter(|| {
                prism_q::run_shots_compiled(&circuit, 10_000, SEED).unwrap();
            });
        });

        let id_lut = format!("lut_only_{}q_10k", n);
        group.bench_function(BenchmarkId::new("lut_only", &id_lut), |b| {
            let mut sampler = prism_q::compile_forward(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_bulk(10_000));
        });

        let id_packed = format!("packed_{}q_10k", n);
        group.bench_function(BenchmarkId::new("packed", &id_packed), |b| {
            let mut sampler = prism_q::compile_forward(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_bulk_packed(10_000));
        });

        let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let id_noisy = format!("noisy_{}q_10k", n);
        group.bench_function(BenchmarkId::new("noisy", &id_noisy), |b| {
            b.iter(|| {
                prism_q::run_shots_noisy(&circuit, &noise, 10_000, SEED).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_compiled_sampler_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_sampler_scale");
    configure_group(&mut group);

    let n = 100;
    let mut circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
    circuit.num_classical_bits = n;
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    for &shots in &[100_000, 1_000_000, 10_000_000] {
        let label = format!("packed_100q_{shots}");
        group.bench_function(BenchmarkId::new("packed", &label), |b| {
            let mut sampler = prism_q::compile_forward(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_bulk_packed(shots));
        });
    }

    group.finish();
}

fn bench_spp(c: &mut Criterion) {
    let mut group = c.benchmark_group("spp");
    configure_group(&mut group);

    let num_samples = 10_000;

    for &t in &[2, 4, 8, 12] {
        let circuit = clifford_t_circuit(10, t, SEED);
        let id = format!("10q_{t}t");

        group.bench_function(BenchmarkId::new("spp_10k", &id), |b| {
            b.iter(|| {
                prism_q::sim::unified_pauli::run_spp(&circuit, num_samples, 42).unwrap();
            });
        });

        if t <= 12 {
            group.bench_function(BenchmarkId::new("spd", &id), |b| {
                b.iter(|| {
                    prism_q::run_spd(&circuit, 0.0, 0).unwrap();
                });
            });
        }
    }

    for &n in &[20, 50, 100] {
        let circuit = clifford_t_circuit(n, 8, SEED);
        let id = format!("{n}q_8t");

        group.bench_function(BenchmarkId::new("spp_10k", &id), |b| {
            b.iter(|| {
                prism_q::sim::unified_pauli::run_spp(&circuit, num_samples, 42).unwrap();
            });
        });
    }

    let circuit_100t = clifford_t_circuit(20, 100, SEED);
    group.bench_function(BenchmarkId::new("spp_10k", "20q_100t"), |b| {
        b.iter(|| {
            prism_q::sim::unified_pauli::run_spp(&circuit_100t, num_samples, 42).unwrap();
        });
    });

    let circuit_1000t = clifford_t_circuit(50, 1000, SEED);
    group.bench_function(BenchmarkId::new("spp_10k", "50q_1000t"), |b| {
        b.iter(|| {
            prism_q::sim::unified_pauli::run_spp(&circuit_1000t, num_samples, 42).unwrap();
        });
    });

    group.finish();
}

fn bench_coalesce_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalesce_baseline");
    configure_group(&mut group);

    let num_samples = 10_000;

    for &(n, depth, t_frac, label) in &[
        (10, 20, 0.05, "10q_d20_t5pct"),
        (10, 50, 0.05, "10q_d50_t5pct"),
        (20, 20, 0.05, "20q_d20_t5pct"),
        (50, 20, 0.05, "50q_d20_t5pct"),
        (100, 10, 0.05, "100q_d10_t5pct"),
    ] {
        let circuit = circuits::clifford_t_circuit(n, depth, t_frac, SEED);

        group.bench_function(BenchmarkId::new("spp_10k", label), |b| {
            b.iter(|| {
                prism_q::sim::unified_pauli::run_spp(&circuit, num_samples, 42).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    // Statevector sweeps
    bench_statevector_random,
    bench_statevector_qft,
    bench_statevector_qft_textbook,
    bench_statevector_qpe,
    bench_statevector_hea,
    bench_statevector_qaoa,
    bench_statevector_clifford,
    bench_statevector_depth_sweep,
    bench_statevector_entanglement,
    bench_statevector_scalability,
    // Stabilizer
    bench_stabilizer_scaling,
    bench_stabilizer_measurement,
    // Factored stabilizer
    bench_factored_stabilizer_scaling,
    bench_factored_stabilizer_local,
    bench_factored_stabilizer_measurement,
    // Sparse
    bench_sparse_scaling,
    bench_sparse_low_entanglement,
    // MPS
    bench_mps_scaling,
    bench_mps_linear_chain,
    bench_mps_hotspots,
    // Product state
    bench_product_scaling,
    // Tensor network
    bench_tn_scaling,
    bench_tn_linear_chain,
    // Auto dispatch
    bench_auto_random,
    bench_auto_qft,
    bench_auto_qft_textbook,
    bench_auto_qpe,
    bench_auto_hea,
    bench_auto_clifford,
    bench_auto_scalability,
    // Cross-backend comparisons
    bench_compare_clifford,
    bench_compare_single_qubit,
    bench_compare_general,
    // Decomposition
    bench_decomposition,
    // Factored backend
    bench_factored_random,
    bench_factored_independent,
    bench_factored_sim_only,
    bench_factored_dynamic,
    bench_factored_dense,
    // Clifford+T (SPD/SPP)
    bench_clifford_t,
    // Stabilizer rank
    bench_stabilizer_rank,
    // Compiled sampler (noiseless + noisy shot sampling)
    bench_compiled_sampler,
    // Compiled sampler at scale (high shot counts)
    bench_compiled_sampler_scale,
    // Stochastic Pauli Propagation (Clifford+T)
    bench_spp,
    // Coalescing baseline (interleaved Clifford+T)
    bench_coalesce_baseline,
);
criterion_main!(benches);
