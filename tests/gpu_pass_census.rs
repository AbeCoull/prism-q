//! Launch, pass and drain counters on a real device. A missing device fails every test
//! unless `PRISM_ALLOW_NO_GPU=1` is set.

#![cfg(all(feature = "gpu", feature = "bench-internal"))]

mod gpu_device;

use std::sync::Arc;

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::circuit::{Circuit, Instruction, recognize_qft_blocks, smallvec};
use prism_q::gates::{BatchPhaseData, Gate};
use prism_q::gpu::{GpuContext, GpuPassCensus};
use prism_q::{StabilizerBackend, StatevectorBackend};

const NUM_QUBITS: usize = 12;

fn census_of(ctx: &Arc<GpuContext>, circuit: &Circuit) -> GpuPassCensus {
    let mut backend = StatevectorBackend::new(42).with_gpu(ctx.clone());
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    let before = ctx.pass_census();
    backend.apply_instructions(&circuit.instructions).unwrap();
    ctx.pass_census() - before
}

fn stabilizer_census_of(ctx: &Arc<GpuContext>, circuit: &Circuit) -> GpuPassCensus {
    let mut backend = StabilizerBackend::new(42).with_gpu(ctx.clone());
    let before = ctx.pass_census();
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    ctx.pass_census() - before
}

#[test]
fn each_gate_kernel_is_one_launch_and_one_pass() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let mut circuit = Circuit::new(NUM_QUBITS, 0);
    for q in 0..NUM_QUBITS {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::Rz(0.1 * q as f64), &[q]);
    }
    for q in 0..NUM_QUBITS - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let gates = circuit.instructions.len() as u64;
    assert_eq!(
        census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: gates,
            passes: gates,
            drains: 0,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn staged_tables_and_measurement_readbacks_drain_the_stream() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let mut circuit = Circuit::new(NUM_QUBITS, 1);
    circuit.add_gate(Gate::H, &[0]);
    let phases = (1..NUM_QUBITS).map(|q| (q, Complex64::from_polar(1.0, 0.3 * q as f64)));
    circuit.instructions.push(Instruction::Gate {
        gate: Gate::BatchPhase(Box::new(BatchPhaseData::new(phases))),
        targets: smallvec![0],
    });
    circuit.add_measure(0, 0);
    // H and the batch phase are one pass each, the batch phase uploads its table, and
    // the measurement reduces (pass plus finalize), reads back, then collapses (pass).
    assert_eq!(
        census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: 5,
            passes: 4,
            drains: 2,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn recognized_qft_runs_as_tiled_passes_with_one_drain() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let expanded =
        prism_q::circuit::expand_qft_blocks(&prism_q::circuits::qft_circuit(20)).into_owned();
    let circuit = recognize_qft_blocks(&expanded).into_owned();
    assert_eq!(circuit.instructions.len(), 1);
    // Stages 11..20 in one gathered pass, 0..11 in the tile pass, then the output
    // bit reversal; the twiddle and phase tables go up once.
    assert_eq!(
        census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: 3,
            passes: 3,
            drains: 1,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn stabilizer_ghz_measure_all_counts_one_readback_chain_per_qubit() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let mut circuit = prism_q::circuits::ghz_circuit(1000);
    circuit.measure_all();
    // Init, then 16 Clifford batches (a flush at each of the 15 word boundaries the CX
    // chain crosses, plus the final one), each staging 4 tables. Qubit 0 measures
    // random (pivot, cascade, fixup); the other 999 are deterministic (pivot, chain).
    // Each pivot and each chain uploads a reset word and reads one back.
    assert_eq!(
        stabilizer_census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: 1 + 16 + 3 + 999 * 2,
            passes: 1 + 16 + 1,
            drains: 16 * 4 + 2 + 999 * 4,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn stabilizer_h_wall_measure_all_is_random_on_every_qubit() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let n = 512;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.measure_all();
    // Init and one Clifford batch, then pivot, cascade and fixup for each qubit.
    assert_eq!(
        stabilizer_census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: 1 + 1 + n as u64 * 3,
            passes: 1 + 1 + n as u64,
            drains: 4 + n as u64 * 2,
            dense_2q: 0,
            tiles: 0,
        }
    );
}
