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
            uploads: 0,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn staged_tables_upload_and_measurement_readbacks_drain_the_stream() {
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
    // H and the batch phase are one pass each and the batch phase uploads its table. The
    // run uploads the pending norm once; the measurement reduces (pass), decides on the
    // device (launch with no pass) and collapses (pass); the run reads the outcome back
    // once at its end.
    assert_eq!(
        census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: 5,
            passes: 4,
            drains: 1,
            uploads: 2,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn recognized_qft_runs_as_tiled_passes_with_one_upload() {
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
            drains: 0,
            uploads: 1,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn stabilizer_ghz_measure_all_runs_one_collapse_on_the_device() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let mut circuit = prism_q::circuits::ghz_circuit(1000);
    circuit.measure_all();
    // Init, then 16 Clifford batches (a flush at each of the 15 word boundaries the CX
    // chain crosses, plus the final one), each uploading 4 tables. The measure run
    // uploads its state once. The first random mask has every bit set, so qubit 0
    // collapses (pivot, cascade, fixup); the second mask is clear, so every stabilizer
    // is diagonal and the other 999 outcomes come from the sign accumulation and its
    // readout, two launches. Drains: two masks and one readback of the run.
    assert_eq!(
        stabilizer_census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: 1 + 16 + 2 + 3 + 2,
            passes: 1 + 16 + 2 + 1 + 1,
            drains: 2 + 1,
            uploads: 16 * 4 + 1,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn stabilizer_h_wall_measure_all_collapses_in_doubling_chunks() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let n = 512;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.measure_all();
    // Init and one Clifford batch (4 table uploads), then the run upload. Every outcome
    // is random: chunks of 1, 2, 4, ..., 256 and a last one of 1 make 10 chunks behind
    // 11 random masks, each mask a launch and a readback. Every qubit collapses
    // (pivot, cascade, fixup). The last mask is clear, so the diagonal finish runs
    // (two launches, nothing left to resolve) before one readback of the run.
    assert_eq!(
        stabilizer_census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: 1 + 1 + 11 + n as u64 * 3 + 2,
            passes: 1 + 1 + 11 + n as u64 + 1,
            drains: 11 + 1,
            uploads: 4 + 1,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn device_sampling_uploads_random_bits_per_chunk() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let n = 128;
    let mut circuit = Circuit::new(n, n);
    for q in 0..12 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.measure_all();
    let mut sampler = prism_q::compile_measurements(&circuit, 42)
        .unwrap()
        .with_gpu(ctx.clone());
    sampler.sample_bulk_packed_device(64).unwrap();
    // Two chunks of 65,536 shots: each uploads its host drawn random bits and launches
    // one sampling kernel; the shots stay on the device. The parity tables went up on
    // the first call.
    let before = ctx.pass_census();
    sampler.sample_bulk_packed_device(131_072).unwrap();
    assert_eq!(
        ctx.pass_census() - before,
        GpuPassCensus {
            launches: 2,
            passes: 0,
            drains: 0,
            uploads: 2,
            dense_2q: 0,
            tiles: 0,
        }
    );
}

#[test]
fn measurements_in_one_run_read_back_once() {
    let Some(ctx) = gpu_device::context() else {
        return;
    };
    let mut circuit = Circuit::new(NUM_QUBITS, 6);
    for q in 0..NUM_QUBITS {
        circuit.add_gate(Gate::H, &[q]);
    }
    for m in 0..6 {
        circuit.add_measure(m, m);
        circuit.add_gate(Gate::Rz(0.2), &[m + 6]);
    }
    let gates = NUM_QUBITS as u64 + 6;
    // Each measurement reduces, decides and collapses on the device; the pending norm
    // goes up once and every outcome comes back in one readback at the end of the run.
    assert_eq!(
        census_of(&ctx, &circuit),
        GpuPassCensus {
            launches: gates + 6 * 3,
            passes: gates + 6 * 2,
            drains: 1,
            uploads: 1,
            dense_2q: 0,
            tiles: 0,
        }
    );
}
