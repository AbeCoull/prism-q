//! Microbenchmarks: gate kernels and small end-to-end simulations.
//!
//! Use `--features bench-fast` for a quick run that reduces warmup and
//! measurement time. Omit for the full suite with default Criterion timing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::BackendKind;
use std::time::Duration;

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

const QUBIT_COUNTS: [usize; 5] = [4, 8, 12, 16, 20];

fn bench_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(BenchmarkId::new("h_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::H, &[0]);
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("rx_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Rx(1.234), &[0]);
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("t_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::T, &[0]);
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
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
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("cz_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Cz, &[0, 1]);
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("swap_gate", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::Swap, &[0, 1]);
                b.iter(|| {
                    sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );
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
                    sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
                });
            },
        );
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
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
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
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
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
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        if n_qubits >= 4 {
            group.bench_with_input(BenchmarkId::new("cccx", n_qubits), &n_qubits, |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::mcu(x_mat, 3), &[0, 1, 2, 3]);
                b.iter(|| {
                    sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
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
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("ry_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::Ry(1.234), &[0]);
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("p_gate", n_qubits), &n_qubits, |b, &n| {
            let mut circuit = Circuit::new(n, 0);
            circuit.add_gate(Gate::P(1.234), &[0]);
            b.iter(|| {
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
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
                    sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
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
                    sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
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
                sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("sxdg_gate", n_qubits),
            &n_qubits,
            |b, &n| {
                let mut circuit = Circuit::new(n, 0);
                circuit.add_gate(Gate::SXdg, &[0]);
                b.iter(|| {
                    sim::run_with(BackendKind::Statevector, &circuit, 42).unwrap();
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
                    sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("classical_only", n_qubits),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    sim::run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_qubit_gates,
    bench_two_qubit_gates,
    bench_measurement,
    bench_qasm_parse_and_simulate,
    bench_high_target_qubit,
    bench_controlled_gates,
    bench_diagonal_parametric_gates,
    bench_cphase_kernel,
    bench_new_gate_types,
    bench_classical_only,
);
criterion_main!(benches);
