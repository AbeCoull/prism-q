//! Launch, pass and drain counters on a real device. Without a usable GPU every test
//! returns before asserting; set `PRISM_REQUIRE_GPU=1` to make that a hard failure.

#![cfg(all(feature = "gpu", feature = "bench-internal"))]

use std::sync::Arc;

use num_complex::Complex64;
use prism_q::StatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::{Circuit, Instruction, recognize_qft_blocks, smallvec};
use prism_q::gates::{BatchPhaseData, Gate};
use prism_q::gpu::{GpuContext, GpuPassCensus};

const NUM_QUBITS: usize = 12;

fn context_or_skip() -> Option<Arc<GpuContext>> {
    match GpuContext::new(0) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU is set but no usable GPU was found ({e}). \
                 Unset it to allow this suite to skip."
            );
            eprintln!("SKIP: no usable GPU ({e})");
            None
        }
    }
}

fn census_of(ctx: &Arc<GpuContext>, circuit: &Circuit) -> GpuPassCensus {
    let mut backend = StatevectorBackend::new(42).with_gpu(ctx.clone());
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    let before = ctx.pass_census();
    backend.apply_instructions(&circuit.instructions).unwrap();
    ctx.pass_census() - before
}

#[test]
fn each_gate_kernel_is_one_launch_and_one_pass() {
    let Some(ctx) = context_or_skip() else {
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
        }
    );
}

#[test]
fn staged_tables_and_measurement_readbacks_drain_the_stream() {
    let Some(ctx) = context_or_skip() else {
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
        }
    );
}

#[test]
fn recognized_qft_runs_as_tiled_passes_with_one_drain() {
    let Some(ctx) = context_or_skip() else {
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
        }
    );
}
