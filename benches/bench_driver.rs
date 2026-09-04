//! Microbenchmarks: gate kernels and small end-to-end simulations.
//!
//! Use `--features bench-fast` for a quick run that reduces warmup and
//! measurement time. Omit for the full suite with default Criterion timing.

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use prism_q::backend::Backend;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{BackendKind, StatevectorBackend};
use std::hint::black_box;

mod common;
use common::{configure_group, run_with};

const QUBIT_COUNTS: [usize; 5] = [4, 8, 12, 16, 20];

/// Time one gate's kernel over a full-superposition `n_qubits` state.
///
/// Two deliberate choices, both of which a row gets wrong by default.
///
/// Applies through [`StatevectorBackend`] rather than `simulate()`. A circuit
/// holding a few gates on a wide register has a largest connected block far
/// narrower than the register, and independent-subsystem decomposition claims
/// such a circuit before a backend is resolved, so the `simulate()` route
/// reports `Decomposed` and never reaches the dense kernel.
///
/// Holds one backend for the whole row rather than rebuilding per iteration.
/// Every gate benched here is unitary, so repeated application walks the same
/// amplitudes and cannot drift the norm, and at 20 qubits a per-iteration
/// rebuild puts a 16 MB allocate-and-zero pass beside every measured one. The
/// state is prepared in full superposition so no amplitude is zero.
fn bench_kernel_row(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    n_qubits: usize,
    gate: Gate,
    targets: &[usize],
) {
    let mut circuit = Circuit::new(n_qubits, 0);
    circuit.add_gate(gate, targets);
    group.bench_function(BenchmarkId::new(name, n_qubits), |b| {
        let mut backend = StatevectorBackend::new(42);
        backend.init(n_qubits, 0).unwrap();
        let mut prep = Circuit::new(n_qubits, 0);
        for q in 0..n_qubits {
            prep.add_gate(Gate::H, &[q]);
        }
        for inst in &prep.instructions {
            backend.apply(inst).unwrap();
        }
        b.iter(|| backend.apply(&circuit.instructions[0]).unwrap());
    });
}

fn bench_single_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_gates");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        bench_kernel_row(&mut group, "h_gate", n_qubits, Gate::H, &[0]);
        bench_kernel_row(&mut group, "rx_gate", n_qubits, Gate::Rx(1.234), &[0]);
        bench_kernel_row(&mut group, "t_gate", n_qubits, Gate::T, &[0]);
    }

    group.finish();
}

fn bench_two_qubit_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gates");
    configure_group(&mut group);

    let cx_4x4 = Gate::Cx.matrix_4x4();

    for &n_qubits in &QUBIT_COUNTS {
        let hi = n_qubits - 1;
        bench_kernel_row(&mut group, "cx_gate", n_qubits, Gate::Cx, &[0, 1]);
        bench_kernel_row(
            &mut group,
            "cx_ctrl_gt_adjacent",
            n_qubits,
            Gate::Cx,
            &[1, 0],
        );
        bench_kernel_row(&mut group, "cx_ctrl_lt_high", n_qubits, Gate::Cx, &[0, hi]);
        bench_kernel_row(&mut group, "cx_ctrl_gt_high", n_qubits, Gate::Cx, &[hi, 0]);
        bench_kernel_row(&mut group, "cz_gate", n_qubits, Gate::Cz, &[0, 1]);
        bench_kernel_row(&mut group, "cz_high", n_qubits, Gate::Cz, &[0, hi]);
        bench_kernel_row(&mut group, "swap_gate", n_qubits, Gate::Swap, &[0, 1]);
        bench_kernel_row(
            &mut group,
            "fused2q_adjacent",
            n_qubits,
            Gate::Fused2q(Box::new(cx_4x4)),
            &[0, 1],
        );
        bench_kernel_row(
            &mut group,
            "fused2q_high",
            n_qubits,
            Gate::Fused2q(Box::new(cx_4x4)),
            &[0, hi],
        );
    }

    group.finish();
}

fn bench_two_qubit_gate_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_qubit_gate_kernels");
    configure_group(&mut group);

    for &n_qubits in &[12, 16, 20, 22] {
        let hi = n_qubits - 1;
        bench_kernel_row(
            &mut group,
            "cx_ctrl_lt_adjacent",
            n_qubits,
            Gate::Cx,
            &[0, 1],
        );
        bench_kernel_row(
            &mut group,
            "cx_ctrl_gt_adjacent",
            n_qubits,
            Gate::Cx,
            &[1, 0],
        );
        bench_kernel_row(&mut group, "cx_ctrl_lt_high", n_qubits, Gate::Cx, &[0, hi]);
        bench_kernel_row(&mut group, "cx_ctrl_gt_high", n_qubits, Gate::Cx, &[hi, 0]);
        bench_kernel_row(&mut group, "cz_adjacent", n_qubits, Gate::Cz, &[0, 1]);
        bench_kernel_row(&mut group, "cz_high", n_qubits, Gate::Cz, &[0, hi]);
    }

    group.finish();
}

fn bench_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("measurement");
    configure_group(&mut group);

    // End-to-end route, not a measure kernel: `simulate()` resolves a backend,
    // allocates, applies and builds an outcome, and an H on one qubit of a wide
    // register decomposes into independent subsystems before the statevector
    // backend is reached. The direct-backend measure rows are `measure_q*_n20`
    // below.
    for &n_qubits in &QUBIT_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("simulate_h_measure", n_qubits),
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
        bench_kernel_row(
            &mut group,
            &format!("h_q{}", target),
            n_qubits,
            Gate::H,
            &[target],
        );
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

// Driven through `StatevectorBackend::apply` rather than `simulate()`, like
// `two_qubit_gate_kernels` above. A single controlled gate leaves every other
// qubit isolated, so the largest connected block is far narrower than the
// register and independent-subsystem decomposition claims the circuit before
// any backend is chosen: routed through `simulate()` these rows resolve to
// `Decomposed` and never reach the kernel they are named for.
fn bench_controlled_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("controlled_gates");
    configure_group(&mut group);

    let h_mat = Gate::H.matrix_2x2();
    let x_mat = Gate::X.matrix_2x2();

    for &n_qubits in &QUBIT_COUNTS {
        bench_kernel_row(&mut group, "cu_h", n_qubits, Gate::cu(h_mat), &[0, 1]);

        if n_qubits >= 3 {
            bench_kernel_row(
                &mut group,
                "toffoli",
                n_qubits,
                Gate::mcu(x_mat, 2),
                &[0, 1, 2],
            );
        }
        if n_qubits >= 4 {
            bench_kernel_row(
                &mut group,
                "cccx",
                n_qubits,
                Gate::mcu(x_mat, 3),
                &[0, 1, 2, 3],
            );
        }
    }

    group.finish();
}

fn bench_diagonal_parametric_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagonal_parametric_gates");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        bench_kernel_row(&mut group, "rz_gate", n_qubits, Gate::Rz(1.234), &[0]);
        bench_kernel_row(&mut group, "ry_gate", n_qubits, Gate::Ry(1.234), &[0]);
        bench_kernel_row(&mut group, "p_gate", n_qubits, Gate::P(1.234), &[0]);
    }

    group.finish();
}

fn bench_cphase_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("cphase_kernel");
    configure_group(&mut group);

    let theta = std::f64::consts::FRAC_PI_4;

    for &n_qubits in &QUBIT_COUNTS {
        bench_kernel_row(
            &mut group,
            "ctrl_lt_target",
            n_qubits,
            Gate::cphase(theta),
            &[0, 1],
        );
        bench_kernel_row(
            &mut group,
            "ctrl_gt_target",
            n_qubits,
            Gate::cphase(theta),
            &[1, 0],
        );
    }

    group.finish();
}

fn bench_new_gate_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_gate_types");
    configure_group(&mut group);

    for &n_qubits in &QUBIT_COUNTS {
        bench_kernel_row(&mut group, "sx_gate", n_qubits, Gate::SX, &[0]);
        bench_kernel_row(&mut group, "sxdg_gate", n_qubits, Gate::SXdg, &[0]);
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
