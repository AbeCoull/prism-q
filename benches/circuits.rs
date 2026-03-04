//! Macrobenchmarks: circuit family sweeps (qubit count and depth).
//!
//! Use `--features bench-fast` for a quick run that reduces warmup and
//! measurement time. Omit for the full suite with default Criterion timing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::BackendKind;
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
            let gate_idx = rng.gen_range(0..cliffords.len());
            circuit.add_gate(cliffords[gate_idx].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n_qubits - 1).step_by(2) {
            if rng.gen_bool(0.5) {
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
            let gate_idx = rng.gen_range(0..gates.len());
            circuit.add_gate(gates[gate_idx].clone(), &[q]);
        }
    }
    circuit
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
    use prism_q::SimOptions;
    let mut group = c.benchmark_group("factored/sim_only_d10");
    configure_group(&mut group);
    let opts = SimOptions::classical_only();

    for &n in &[16, 20, 24] {
        let circuit = circuits::random_circuit(n, 10, SEED);

        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with_opts(BackendKind::Statevector, circ, 42, opts.clone()).unwrap()
            });
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| sim::run_with_opts(BackendKind::Factored, circ, 42, opts.clone()).unwrap());
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
    // Sparse
    bench_sparse_scaling,
    bench_sparse_low_entanglement,
    // MPS
    bench_mps_scaling,
    bench_mps_linear_chain,
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
);
criterion_main!(benches);
