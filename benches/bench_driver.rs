//! Microbenchmarks: gate kernels and small end-to-end simulations.
//!
//! Use `--features bench-fast` for a quick run that reduces warmup and
//! measurement time. Omit for the full suite with default Criterion timing.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use prism_q::backend::Backend;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{BackendKind, StatevectorBackend};
use std::hint::black_box;

mod common;
use common::{configure_group, run_with};

const QUBIT_COUNTS: [usize; 5] = [4, 8, 12, 16, 20];

fn bench_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(BenchmarkId::new("h_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::H, &[0]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("rx_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Rx(1.234), &[0]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("t_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::T, &[0]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gates");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(BenchmarkId::new("cx_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Cx, &[0, 1]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("cx_ctrl_gt_adjacent", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::Cx, &[1, 0]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cx_ctrl_lt_high", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::Cx, &[0, n - 1]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cx_ctrl_gt_high", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::Cx, &[n - 1, 0]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("cz_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Cz, &[0, 1]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("cz_high", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Cz, &[0, n - 1]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("swap_gate", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::Swap, &[0, 1]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fused2q_adjacent", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())), &[0, 1]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fused2q_high", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())), &[0, n - 1]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_two_qubit_gate_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gate_kernels");
    configure_group(&mut group);

    for &n_qubits in &[12, 16, 20, 22] {
        let mut cx_lt_adj = Circuit::new(n_qubits, 0);
        cx_lt_adj.add_gate(Gate::Cx, &[0, 1]);
        group.bench_function(BenchmarkId::new("cx_ctrl_lt_adjacent", n_qubits), |b| {
            b.iter_batched(
                || {
                    let mut backend = StatevectorBackend::new(42);
                    backend.init(n_qubits, 0).unwrap();
                    backend
                },
                |mut backend| {
                    backend.apply(&cx_lt_adj.instructions[0]).unwrap();
                    black_box(backend.state_vector());
                },
                BatchSize::SmallInput,
            );
        });

        let mut cx_gt_adj = Circuit::new(n_qubits, 0);
        cx_gt_adj.add_gate(Gate::Cx, &[1, 0]);
        group.bench_function(BenchmarkId::new("cx_ctrl_gt_adjacent", n_qubits), |b| {
            b.iter_batched(
                || {
                    let mut backend = StatevectorBackend::new(42);
                    backend.init(n_qubits, 0).unwrap();
                    backend
                },
                |mut backend| {
                    backend.apply(&cx_gt_adj.instructions[0]).unwrap();
                    black_box(backend.state_vector());
                },
                BatchSize::SmallInput,
            );
        });

        let mut cx_lt_high = Circuit::new(n_qubits, 0);
        cx_lt_high.add_gate(Gate::Cx, &[0, n_qubits - 1]);
        group.bench_function(BenchmarkId::new("cx_ctrl_lt_high", n_qubits), |b| {
            b.iter_batched(
                || {
                    let mut backend = StatevectorBackend::new(42);
                    backend.init(n_qubits, 0).unwrap();
                    backend
                },
                |mut backend| {
                    backend.apply(&cx_lt_high.instructions[0]).unwrap();
                    black_box(backend.state_vector());
                },
                BatchSize::SmallInput,
            );
        });

        let mut cx_gt_high = Circuit::new(n_qubits, 0);
        cx_gt_high.add_gate(Gate::Cx, &[n_qubits - 1, 0]);
        group.bench_function(BenchmarkId::new("cx_ctrl_gt_high", n_qubits), |b| {
            b.iter_batched(
                || {
                    let mut backend = StatevectorBackend::new(42);
                    backend.init(n_qubits, 0).unwrap();
                    backend
                },
                |mut backend| {
                    backend.apply(&cx_gt_high.instructions[0]).unwrap();
                    black_box(backend.state_vector());
                },
                BatchSize::SmallInput,
            );
        });

        let mut cz_adj = Circuit::new(n_qubits, 0);
        cz_adj.add_gate(Gate::Cz, &[0, 1]);
        group.bench_function(BenchmarkId::new("cz_adjacent", n_qubits), |b| {
            b.iter_batched(
                || {
                    let mut backend = StatevectorBackend::new(42);
                    backend.init(n_qubits, 0).unwrap();
                    backend
                },
                |mut backend| {
                    backend.apply(&cz_adj.instructions[0]).unwrap();
                    black_box(backend.state_vector());
                },
                BatchSize::SmallInput,
            );
        });

        let mut cz_high = Circuit::new(n_qubits, 0);
        cz_high.add_gate(Gate::Cz, &[0, n_qubits - 1]);
        group.bench_function(BenchmarkId::new("cz_high", n_qubits), |b| {
            b.iter_batched(
                || {
                    let mut backend = StatevectorBackend::new(42);
                    backend.init(n_qubits, 0).unwrap();
                    backend
                },
                |mut backend| {
                    backend.apply(&cz_high.instructions[0]).unwrap();
                    black_box(backend.state_vector());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("measurement");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("measure_superposition", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 1);
                circuit.add_gate(Gate::H, &[0]);
                circuit.add_measure(0, 0);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );
    }

    // The measurement family chunks by 2^(qubit+1), so a top-qubit target is one
    // chunk over the whole state where qubit 0 is 2^(n-1) of them. Both rows walk
    // the same 2^n amplitudes; only the chunking differs, so q0 is the control.
    //
    // The read and collapse rows hold one backend for the whole row: the reads
    // do not mutate, and a repeated collapse on an already collapsed state walks
    // the same amplitudes. Rebuilding a 16 MB backend per iteration would put an
    // allocate-and-zero pass beside every measured one.
    const N: usize = 20;
    let hi = N - 1;

    let prepared = |target: usize, classical: usize| {
        let mut backend = StatevectorBackend::new(42);
        backend.init(N, classical).unwrap();
        let mut prep = Circuit::new(N, classical);
        prep.add_gate(Gate::X, &[target]);
        backend.apply(&prep.instructions[0]).unwrap();
        backend
    };

    for &(target, label) in &[(0usize, "prob_one_q0_n20"), (hi, "prob_one_q19_n20")] {
        group.bench_function(label, |b| {
            let backend = prepared(target, 0);
            b.iter(|| black_box(backend.qubit_probability(target).unwrap()));
        });
    }

    for &(target, label) in &[(0usize, "rdm_q0_n20"), (hi, "rdm_q19_n20")] {
        group.bench_function(label, |b| {
            let backend = prepared(target, 0);
            b.iter(|| black_box(backend.reduced_density_matrix_1q(target).unwrap()));
        });
    }

    for &(target, label) in &[(0usize, "measure_q0_n20"), (hi, "measure_q19_n20")] {
        let measure = prism_q::Instruction::Measure {
            qubit: target,
            classical_bit: 0,
        };
        group.bench_function(label, |b| {
            let mut backend = prepared(target, 1);
            b.iter(|| backend.apply(&measure).unwrap());
        });
    }

    // Reset is the one row that cannot reuse its state: once the qubit is |0>
    // the fold stops copying, so each iteration needs a state with weight on the
    // |1> branch. `PerIteration` keeps that to one live backend at a time.
    for &(target, label) in &[(0usize, "reset_q0_n20"), (hi, "reset_q19_n20")] {
        let mut circuit = Circuit::new(N, 0);
        circuit.add_reset(target);
        group.bench_function(label, |b| {
            b.iter_batched(
                || prepared(target, 0),
                |mut backend| {
                    backend.apply(&circuit.instructions[0]).unwrap();
                    black_box(backend.state_vector());
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

fn bench_qasm_parse_and_simulate(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_qasm");
    configure_group(&mut group);

    let bell_qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;

    group.bench_function("bell_state", |b| {
        b.iter(|| {
            prism_q::run_qasm(bell_qasm, 42).unwrap();
        });
    });

    let ghz_5_qasm = r#"
        OPENQASM 3.0;
        qubit[5] q;
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        cx q[2], q[3];
        cx q[3], q[4];
    "#;

    group.bench_function("ghz_5", |b| {
        b.iter(|| {
            prism_q::run_qasm(ghz_5_qasm, 42).unwrap();
        });
    });

    group.finish();
}

fn bench_high_target_qubit(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_target_qubit");
    configure_group(&mut group);

    for &(n_qubits, target) in &[(16, 13), (20, 15), (20, 17)] {
        let label = format!("h_q{}_n{}", target, n_qubits);
        group.bench_function(&label, |b| {
            let mut circuit = Circuit::new(n_qubits, 0);
            circuit.add_gate(Gate::H, &[target]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    // Diagonal counterpart of the rows above. The diagonal pass tiles by
    // 2^(target+1), so at n=20 target 19 leaves one tile and target 17 leaves
    // four, while the general 1q kernel splits either. Rz scales both halves,
    // where P leaves the |0> half alone. Rz is unitary, so one backend serves
    // the whole row and no allocation lands beside a measured pass.
    for &(n_qubits, target) in &[(20, 17), (20, 19)] {
        let mut circuit = Circuit::new(n_qubits, 0);
        circuit.add_gate(Gate::Rz(0.7), &[target]);
        group.bench_function(format!("rz_q{}_n{}", target, n_qubits), |b| {
            let mut backend = StatevectorBackend::new(42);
            backend.init(n_qubits, 0).unwrap();
            b.iter(|| backend.apply(&circuit.instructions[0]).unwrap());
        });
    }

    group.finish();
}

fn bench_controlled_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("controlled_gates");
    configure_group(&mut group);

    let h_mat = Gate::H.matrix_2x2();

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(BenchmarkId::new("cu_h", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::cu(h_mat), &[0, 1]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    let x_mat = Gate::X.matrix_2x2();
    for &n_qubits in &[4, 8, 12, 16, 20] {
        if n_qubits < 3 {
            continue;
        }
        group.bench_with_input(BenchmarkId::new("toffoli", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::mcu(x_mat, 2), &[0, 1, 2]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        if n_qubits >= 4 {
            group.bench_with_input(BenchmarkId::new("cccx", n_qubits), &n_qubits, |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::mcu(x_mat, 3), &[0, 1, 2, 3]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            });
        }
    }

    group.finish();
}

fn bench_diagonal_parametric_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagonal_parametric_gates");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(BenchmarkId::new("rz_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Rz(1.234), &[0]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("ry_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Ry(1.234), &[0]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("p_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::P(1.234), &[0]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_cphase_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("cphase_kernel");
    configure_group(&mut group);

    let theta = std::f64::consts::FRAC_PI_4;

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("ctrl_lt_target", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::cphase(theta), &[0, 1]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ctrl_gt_target", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::cphase(theta), &[1, 0]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_new_gate_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_gate_types");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(BenchmarkId::new("sx_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::SX, &[0]);
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("sxdg_gate", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::SXdg, &[0]);
                b.iter(|| {
                    run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_classical_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("classical_only");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        let mut circuit = Circuit::new(n_qubits, 1);
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
        }
        for q in 0..n_qubits - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
        circuit.add_measure(0, 0);

        group.bench_with_input(
            BenchmarkId::new("with_probs", n_qubits),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("classical_only", n_qubits),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets =
        bench_single_qubit_gates,
    bench_two_qubit_gates,
    bench_two_qubit_gate_kernels,
    bench_measurement,
    bench_qasm_parse_and_simulate,
    bench_high_target_qubit,
    bench_controlled_gates,
    bench_diagonal_parametric_gates,
    bench_cphase_kernel,
    bench_new_gate_types,
    bench_classical_only
}
criterion_main!(benches);
