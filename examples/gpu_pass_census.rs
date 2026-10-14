//! Kernel launches, state passes and stream drains per `gpu/throughput` row.
//!
//! Counts only, no timing: each row expands and fuses the circuit as `simulate` does for
//! the device backend, applies it once to a resident state, and prints the counter
//! difference. A width that does not fit free VRAM prints as skipped.
//!
//! Build + run (`cargo run --release --features "parallel gpu bench-internal" --example gpu_pass_census`).

use std::borrow::Cow;

use prism_q::StatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::fusion::fuse_circuit_for_profile;
use prism_q::circuit::{Circuit, expand_pauli_rotations, expand_qft_blocks, recognize_qft_blocks};
use prism_q::circuits;
use prism_q::gpu::GpuContext;

const SEED: u64 = 0xDEAD_BEEF;
const SIZES: [usize; 5] = [20, 22, 24, 26, 28];
const CIRCUITS: [&str; 4] = ["qft", "hea_l5", "qv", "random_d10"];

fn build(name: &str, n: usize) -> Circuit {
    match name {
        "qft" => circuits::qft_circuit(n),
        "hea_l5" => circuits::hardware_efficient_ansatz(n, 5, SEED),
        "qv" => circuits::quantum_volume_circuit(n, n, SEED),
        "random_d10" => circuits::random_circuit(n, 10, SEED),
        _ => unreachable!("unknown circuit {name}"),
    }
}

fn fuse_for(backend: &StatevectorBackend, circuit: &Circuit) -> Circuit {
    let blocks = if backend.supports_qft_block() {
        recognize_qft_blocks(circuit)
    } else {
        expand_qft_blocks(circuit)
    };
    let rotations = if backend.supports_pauli_rotation() {
        Cow::Borrowed(&*blocks)
    } else {
        expand_pauli_rotations(&blocks)
    };
    fuse_circuit_for_profile(
        &rotations,
        backend.supports_fused_gates(),
        backend.fusion_state_qubits(circuit.num_qubits),
        backend.fusion_profile(circuit.num_qubits),
    )
    .into_owned()
}

fn main() {
    let ctx = GpuContext::new(0).expect("no usable GPU");
    println!(
        "{:<12} {:>3} {:>7} {:>7} {:>9} {:>7} {:>7} {:>8} {:>6}",
        "circuit", "n", "gates", "fused", "launches", "passes", "drains", "dense2q", "tiles"
    );
    for name in CIRCUITS {
        for n in SIZES {
            let circuit = build(name, n);
            if !ctx.fits_statevector_with_scratch(n).unwrap_or(false) {
                println!("{name:<12} {n:>3} skipped: does not fit free VRAM");
                continue;
            }
            let mut backend = StatevectorBackend::new(42).with_gpu(ctx.clone());
            backend.init(n, 0).unwrap();
            let fused = fuse_for(&backend, &circuit);
            let before = ctx.pass_census();
            backend.apply_instructions(&fused.instructions).unwrap();
            ctx.synchronize().unwrap();
            let census = ctx.pass_census() - before;
            println!(
                "{:<12} {:>3} {:>7} {:>7} {:>9} {:>7} {:>7} {:>8} {:>6}",
                name,
                n,
                circuit.instructions.len(),
                fused.instructions.len(),
                census.launches,
                census.passes,
                census.drains,
                census.dense_2q,
                census.tiles
            );
        }
    }
}
