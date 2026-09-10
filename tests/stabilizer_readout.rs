//! Readout error through the public entry points of the Clifford samplers:
//! the compiled aggregates, the block-filtered compile path, and the per-shot
//! replay fallback. Route-sensitive coverage lives in the crate's own noise
//! tests, which can pin the frame and compiled selection.

use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{NoiseModel, ResolvedBackend, compile_noisy, run_shots_noisy, sim};

const SEED: u64 = 42;
const SHOTS: usize = 20_000;
/// Five sigma on a rate estimated from [`SHOTS`] draws, rounded up.
const RATE_EPS: f64 = 0.02;

/// `X q0; X q1` on three qubits, so the record reads `1, 1, 0` with no noise.
fn deterministic_circuit() -> Circuit {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);
    circuit
}

fn asymmetric_model(circuit: &Circuit) -> NoiseModel {
    let mut noise = NoiseModel::uniform_depolarizing(circuit, 0.0);
    noise.set_bit_readout_error(0, 0.40, 0.05);
    noise.set_bit_readout_error(2, 0.20, 0.60);
    noise
}

fn rate(shots: &[Vec<bool>], bit: usize, want: bool) -> f64 {
    shots.iter().filter(|s| s[bit] == want).count() as f64 / shots.len() as f64
}

#[test]
fn a_classical_bit_no_measurement_writes_is_never_flipped() {
    let mut circuit = Circuit::new(2, 3);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.set_bit_readout_error(0, 0.0, 0.30);
    noise.set_bit_readout_error(2, 0.8, 0.8);
    let result = sim::simulate(&circuit)
        .noise(&noise)
        .seed(SEED)
        .shots(SHOTS)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::CompiledStabilizer);

    assert!(
        result.shots.iter().all(|s| !s[2]),
        "bit 2 holds no outcome to misread and must stay unwritten"
    );
    assert!(
        result.shots.iter().all(|s| !s[1]),
        "bit 1 carries no readout entry and must keep its noiseless value"
    );
    // The entry on the unwritten bit must not consume the draws the written
    // record needs, so bit 0 still reads its own rate.
    let flipped = rate(&result.shots, 0, false);
    assert!(
        (flipped - 0.30).abs() < RATE_EPS,
        "bit 0 takes p10 = 0.30, got {flipped}"
    );
}

#[test]
fn compiled_sampler_counts_and_marginals_carry_readout() {
    let circuit = deterministic_circuit();
    let noise = asymmetric_model(&circuit);

    let mut sampler = compile_noisy(&circuit, &noise, SEED).unwrap();
    let marginals = sampler.sample_marginals(SHOTS);
    assert!(
        (marginals[0] - 0.95).abs() < RATE_EPS,
        "record 0 reads 1 unless p10 = 0.05 fires, got {}",
        marginals[0]
    );
    assert_eq!(marginals[1], 1.0);
    assert!(
        (marginals[2] - 0.20).abs() < RATE_EPS,
        "record 2 reads 1 when p01 = 0.20 fires, got {}",
        marginals[2]
    );

    let mut sampler = compile_noisy(&circuit, &noise, SEED).unwrap();
    let counts = sampler.sample_counts(SHOTS);
    let total: u64 = counts.values().sum();
    assert_eq!(total, SHOTS as u64);
    let record_two_set: u64 = counts
        .iter()
        .filter(|(key, _)| key[0] & 0b100 != 0)
        .map(|(_, n)| n)
        .sum();
    let observed = record_two_set as f64 / total as f64;
    assert!(
        (observed - 0.20).abs() < RATE_EPS,
        "record 2 reads 1 when p01 = 0.20 fires, got {observed}"
    );
}

// Two independent two-qubit blocks over four qubits, which is what
// `compile_noisy` splits on. The block-filtered compile builds its own sampler
// and would otherwise carry an empty readout table.
#[test]
fn the_block_filtered_compile_carries_readout() {
    let mut circuit = Circuit::new(4, 4);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    for bit in 0..4 {
        circuit.add_measure(bit, bit);
    }

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.set_bit_readout_error(0, 0.0, 0.25);
    noise.set_bit_readout_error(3, 0.35, 0.0);

    let mut sampler = compile_noisy(&circuit, &noise, SEED).unwrap();
    let marginals = sampler.sample_marginals(SHOTS);
    assert!(
        (marginals[0] - 0.75).abs() < RATE_EPS,
        "record 0 reads 1 unless p10 = 0.25 fires, got {}",
        marginals[0]
    );
    assert_eq!(marginals[1], 1.0);
    assert_eq!(marginals[2], 0.0);
    assert!(
        (marginals[3] - 0.35).abs() < RATE_EPS,
        "record 3 reads 1 when p01 = 0.35 fires, got {}",
        marginals[3]
    );
}

// A mid-circuit measurement puts `run_shots_noisy` on per-shot replay, the one
// route that applies readout to the unpacked record, on the shot's own stream.
#[test]
fn the_replay_route_carries_readout() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_measure(1, 1);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.set_bit_readout_error(0, 0.0, 0.25);

    let result = run_shots_noisy(&circuit, &noise, SHOTS, SEED).unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::Stabilizer);
    let flipped = rate(&result.shots, 0, false);
    assert!(
        (flipped - 0.25).abs() < RATE_EPS,
        "bit 0 takes p10 = 0.25, got {flipped}"
    );
    assert!(
        result.shots.iter().all(|s| s[1]),
        "bit 1 carries no readout entry and must keep its noiseless value"
    );
}

#[test]
fn readout_draws_are_reproducible_across_runs() {
    let circuit = deterministic_circuit();
    let noise = asymmetric_model(&circuit);
    let run = |seed: u64| {
        sim::simulate(&circuit)
            .noise(&noise)
            .seed(seed)
            .shots(4_000)
            .unwrap()
    };

    let first = run(SEED);
    assert_eq!(first.metadata.backend, ResolvedBackend::CompiledStabilizer);
    assert_eq!(first.shots, run(SEED).shots);
    assert_ne!(first.shots, run(SEED + 1).shots);
}
