//! The two-qubit depolarizing channel through the public entry points of the
//! Clifford samplers: the auto route, the compiled aggregates, the
//! block-filtered compile, and the per-shot replay fallback. Route-sensitive
//! coverage lives in the crate's own noise tests.

use prism_q::circuit::{Circuit, Instruction};
use prism_q::gates::Gate;
use prism_q::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel};
use prism_q::{
    BackendKind, Engine, PauliTerm, compile_noisy, density_matrix_expectation_values,
    noisy_marginals_analytical, run_shots_homological, run_shots_noisy, simulate,
};

const SEED: u64 = 42;
const SHOTS: usize = 20_000;
/// Five sigma on a Z expectation estimated from [`SHOTS`] draws at the
/// worst-case variance, rounded up.
const BAND: f64 = 0.036;

/// `X q0` then a CX chain, so the ideal record is all ones and every
/// single-qubit Z moves with the noise. `pad` is a run of Z on qubit 0, which
/// no Z-basis outcome sees and which raises the gate count per qubit past the
/// cutoff that would otherwise send the model to the frame sampler.
fn ones_chain(n: usize, pad: usize) -> Circuit {
    let mut circuit = Circuit::new(n, n);
    circuit.add_gate(Gate::X, &[0]);
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for _ in 0..pad {
        circuit.add_gate(Gate::Z, &[0]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    circuit
}

fn pair_noise(circuit: &Circuit, p: f64) -> NoiseModel {
    let mut after_gate = vec![Vec::new(); circuit.instructions.len()];
    for (slot, inst) in after_gate.iter_mut().zip(&circuit.instructions) {
        if let Instruction::Gate { targets, .. } = inst {
            if targets.len() == 2 {
                slot.push(NoiseEvent {
                    channel: NoiseChannel::TwoQubitDepolarizing { p },
                    qubits: targets.iter().copied().collect(),
                });
            }
        }
    }
    NoiseModel {
        after_gate,
        readout: vec![None; circuit.num_classical_bits],
    }
}

/// Every single-qubit Z, then every nearest-neighbour ZZ.
fn z_observables(n: usize) -> Vec<Vec<PauliTerm>> {
    let mut observables: Vec<Vec<PauliTerm>> = (0..n).map(|q| vec![PauliTerm::z(q)]).collect();
    observables.extend((0..n - 1).map(|q| vec![PauliTerm::z(q), PauliTerm::z(q + 1)]));
    observables
}

fn z_expectations(shots: &[Vec<bool>], n: usize) -> Vec<f64> {
    let total = shots.len() as f64;
    let mut out: Vec<f64> = (0..n)
        .map(|q| 1.0 - 2.0 * shots.iter().filter(|s| s[q]).count() as f64 / total)
        .collect();
    out.extend(
        (0..n - 1)
            .map(|q| 1.0 - 2.0 * shots.iter().filter(|s| s[q] != s[q + 1]).count() as f64 / total),
    );
    out
}

fn assert_within_band(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < BAND,
            "{what} observable {i}: sampled {g:.4} against density matrix {w:.4}"
        );
    }
}

// Each qubit collects one letter from every event that reaches it, so its Z
// expectation separates the joint draw from an implementation resolving only
// the first target of a pair. The nearest-neighbour parities cannot: each is
// fed by one letter of each of two events, so dropping the second leaves the
// same marginal. The record is deterministic without noise, which is what
// keeps all fifteen observables able to move.
#[test]
fn chain_correlators_match_the_density_matrix() {
    let n = 8;
    let circuit = ones_chain(n, 18);
    let noise = pair_noise(&circuit, 0.05);

    let result = simulate(&circuit)
        .noise(&noise)
        .seed(SEED)
        .shots(SHOTS)
        .unwrap();
    assert_eq!(result.metadata.engine, Some(Engine::NoisyCompiledSampler));

    let want =
        density_matrix_expectation_values(&circuit, &z_observables(n), Some(&noise), SEED).unwrap();
    assert_within_band(&z_expectations(&result.shots, n), &want, "ones chain");
}

// `sample_counts` and `sample_marginals` reduce from their own buffers rather
// than from the materialized shots, so each needs its own reading of the
// channel.
#[test]
fn the_compiled_aggregates_carry_the_pair_channel() {
    let n = 8;
    let circuit = ones_chain(n, 18);
    let noise = pair_noise(&circuit, 0.05);
    let want =
        density_matrix_expectation_values(&circuit, &z_observables(n), Some(&noise), SEED).unwrap();

    let mut sampler = compile_noisy(&circuit, &noise, SEED).unwrap();
    let marginals = sampler.sample_marginals(SHOTS);
    for (q, one_rate) in marginals.iter().enumerate() {
        assert!(
            (1.0 - 2.0 * one_rate - want[q]).abs() < BAND,
            "marginal {q}: sampled {:.4} against density matrix {:.4}",
            1.0 - 2.0 * one_rate,
            want[q]
        );
    }

    let mut sampler = compile_noisy(&circuit, &noise, SEED).unwrap();
    let counts = sampler.sample_counts(SHOTS);
    let mut shots: Vec<Vec<bool>> = Vec::with_capacity(SHOTS);
    for (key, count) in &counts {
        let record: Vec<bool> = (0..n).map(|q| (key[q / 64] >> (q % 64)) & 1 != 0).collect();
        shots.extend(std::iter::repeat_n(record, *count as usize));
    }
    assert_eq!(shots.len(), SHOTS);
    assert_within_band(&z_expectations(&shots, n), &want, "counts");
}

// The block-filtered compile holds one block's propagated masks at a time and
// would read the second qubit of a straddling pair through the other block's
// local indices, leaving qubit 2 noiseless and qubit 0 doubly noisy.
#[test]
fn a_block_straddling_pair_keeps_both_qubits_noisy() {
    let mut circuit = Circuit::new(4, 4);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::X, &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    for q in 0..4 {
        circuit.add_measure(q, q);
    }

    let mut noise = pair_noise(&circuit, 0.0);
    noise.after_gate[1] = vec![NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.5 },
        qubits: [1, 2].into_iter().collect(),
    }];

    let observables: Vec<Vec<PauliTerm>> = (0..4)
        .map(|q| vec![PauliTerm::z(q)])
        .chain(std::iter::once(vec![PauliTerm::z(1), PauliTerm::z(2)]))
        .collect();
    let want =
        density_matrix_expectation_values(&circuit, &observables, Some(&noise), SEED).unwrap();

    let mut sampler = compile_noisy(&circuit, &noise, SEED).unwrap();
    let shots = sampler.sample_bulk_packed(SHOTS).to_shots();
    let total = SHOTS as f64;
    let mut got: Vec<f64> = (0..4)
        .map(|q| 1.0 - 2.0 * shots.iter().filter(|s| s[q]).count() as f64 / total)
        .collect();
    got.push(1.0 - 2.0 * shots.iter().filter(|s| s[1] != s[2]).count() as f64 / total);
    assert_within_band(&got, &want, "straddling pair");
}

/// Shots for the one comparison against an independently written sampler.
/// Five sigma on the difference of two Z expectations at this count is 0.016,
/// under the 0.027 that a sixteen-product branch convention on one side would
/// move the first measured bit by.
const REPLAY_SHOTS: usize = 200_000;
const REPLAY_BAND: f64 = 0.016;

// A mid-circuit measurement takes the per-shot replay fallback, which holds no
// packed record and applies the branch as gates on the backend. The trajectory
// engine draws the same channel from a table written years apart from this
// one, so agreement pins the branch convention rather than the arithmetic.
#[test]
fn the_replay_fallback_matches_the_trajectory_engine() {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);
    let noise = pair_noise(&circuit, 0.4);

    let replay = run_shots_noisy(&circuit, &noise, REPLAY_SHOTS, SEED).unwrap();
    let trajectory = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(REPLAY_SHOTS)
        .unwrap();

    let replay_z = z_expectations(&replay.shots, 3);
    let trajectory_z = z_expectations(&trajectory.shots, 3);
    for (i, (a, b)) in replay_z.iter().zip(&trajectory_z).enumerate() {
        assert!(
            (a - b).abs() < REPLAY_BAND,
            "observable {i}: replay {a:.4} against trajectory {b:.4}"
        );
    }
}

// The chain complex holds one column per single-qubit error, so a live pair
// has no column to occupy. `run_shots_noisy` still answers: the homological
// attempt is a fallible probe and the compiled family takes over.
#[test]
fn the_homological_sampler_rejects_a_live_pair_channel() {
    let circuit = ones_chain(4, 0);
    let noise = pair_noise(&circuit, 0.02);

    assert!(!noise.is_pauli_only());
    assert!(noise.ensure_pauli_only().is_err());
    assert!(run_shots_homological(&circuit, &noise, 1000, SEED).is_err());
    assert!(noisy_marginals_analytical(&circuit, &noise, SEED).is_err());
    assert!(run_shots_noisy(&circuit, &noise, 1000, SEED).is_ok());

    let inert = pair_noise(&circuit, 0.0);
    assert!(inert.is_pauli_only());
    assert!(run_shots_homological(&circuit, &inert, 1000, SEED).is_ok());
    assert!(noisy_marginals_analytical(&circuit, &inert, SEED).is_ok());
}
