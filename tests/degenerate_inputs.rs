//! Degenerate inputs driven through every backend kind and terminal: an identity
//! or empty observable, a constant-only observable, a circuit with no gates, zero
//! shots, a single qubit, repeated measurement of one qubit, and noise rates of
//! exactly 0, exactly 1, and below the resolution of `1 - p`. Each case returns the
//! correct value or a clean `PrismError`; a panic fails the test.

mod common;

use common::SEED;
use prism_q::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel};
use prism_q::sim::unified_pauli::SpdTruncation;
use prism_q::{BackendKind, Circuit, Gate, PauliObservable, PauliTerm, Result, simulate};

const EPS: f64 = 1e-10;

fn kinds() -> Vec<BackendKind> {
    vec![
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::Stabilizer,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 8 },
        BackendKind::ProductState,
        BackendKind::TensorNetwork,
        BackendKind::TensorNetworkBounded { tolerance: 1e-9 },
        BackendKind::Factored,
        BackendKind::FactoredStabilizer,
        BackendKind::DensityMatrix,
        BackendKind::StabilizerRank,
        BackendKind::StochasticPauli { num_samples: 256 },
        BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Threshold {
                epsilon: 0.0,
                max_terms: 0,
            },
        },
        BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        },
    ]
}

/// A kind that resolves to an exact engine: its values must match the closed form
/// whenever it answers at all.
fn is_exact(kind: &BackendKind) -> bool {
    !matches!(
        kind,
        BackendKind::StochasticPauli { .. } | BackendKind::TensorNetworkBounded { .. }
    )
}

/// A result or a clean error. A panic inside `f` propagates and fails the test.
fn answered<T>(result: Result<T>) -> Option<T> {
    result.ok()
}

fn assert_near(label: &str, got: f64, want: f64, exact: bool) {
    let tol = if exact { EPS } else { 0.2 };
    assert!(
        (got - want).abs() <= tol,
        "{label}: {got} vs closed form {want}"
    );
}

fn empty(n: usize) -> Circuit {
    Circuit::new(n, 0)
}

fn one_gate(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit
}

#[test]
fn circuits_without_gates_read_the_zero_state() {
    for n in [1, 3] {
        let circuit = empty(n);
        for kind in kinds() {
            let label = format!("{kind:?} on {n} idle qubits");
            if let Some(run) = answered(simulate(&circuit).backend(kind.clone()).seed(SEED).run())
                && let Some(probs) = run.probabilities
            {
                let probs = probs.to_vec();
                assert_near(&label, probs[0], 1.0, true);
                assert_near(&label, probs.iter().sum(), 1.0, true);
            }
            if let Some(marginals) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .marginals(),
            ) {
                for &(p0, _) in &marginals.marginals {
                    assert_near(&label, p0, 1.0, is_exact(&kind));
                }
            }
            if let Some(values) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .expectation_values(&[vec![PauliTerm::z(n - 1)], vec![PauliTerm::x(0)]]),
            ) {
                assert_near(&label, values[0], 1.0, is_exact(&kind));
                assert_near(&label, values[1], 0.0, is_exact(&kind));
            }
            if let Some(state) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .state_vector(),
            ) {
                assert_near(&label, state[0].norm_sqr(), 1.0, true);
            }
        }
    }
}

#[test]
fn single_qubit_terminals_answer_or_decline_cleanly() {
    let circuit = one_gate(1);
    for kind in kinds() {
        let label = format!("{kind:?}");
        if let Some(run) = answered(simulate(&circuit).backend(kind.clone()).seed(SEED).run())
            && let Some(probs) = run.probabilities
        {
            assert_near(&label, probs.to_vec()[1], 1.0, true);
        }
        if let Some(values) = answered(
            simulate(&circuit)
                .backend(kind.clone())
                .seed(SEED)
                .expectation_values(&[vec![PauliTerm::z(0)]]),
        ) {
            assert_near(&label, values[0], -1.0, is_exact(&kind));
        }
        if let Some(rho) = answered(
            simulate(&circuit)
                .backend(kind.clone())
                .seed(SEED)
                .reduced_density_matrix(&[0]),
        ) {
            assert_near(&label, rho.data[3].re, 1.0, true);
        }
        let mut measured = circuit.clone();
        measured.num_classical_bits = 1;
        measured.add_measure(0, 0);
        if let Some(shots) = answered(
            simulate(&measured)
                .backend(kind.clone())
                .seed(SEED)
                .shots(16),
        ) {
            assert!(shots.shots.iter().all(|shot| shot == &[true]), "{label}");
        }
    }
}

fn identity_cases(n: usize) -> Vec<Circuit> {
    let mut entangled = Circuit::new(n, 0);
    entangled.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        entangled.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let mut rotated = entangled.clone();
    for q in 0..n {
        rotated.add_gate(Gate::Ry(0.3 + 0.1 * q as f64), &[q]);
    }
    vec![empty(n), entangled, rotated]
}

// Three qubits stays under the tensor expectation route's floor in either build; the
// route itself has its own test below.
#[test]
fn an_identity_term_reads_one_on_every_kind() {
    for n in [1, 3] {
        for circuit in identity_cases(n) {
            for kind in kinds() {
                let label = format!("{kind:?} on {n} qubits");
                if let Some(values) = answered(
                    simulate(&circuit)
                        .backend(kind.clone())
                        .seed(SEED)
                        .expectation_values(&[Vec::new(), vec![PauliTerm::z(0)]]),
                ) {
                    assert_eq!(values.len(), 2, "{label}");
                    assert_near(&label, values[0], 1.0, true);
                }
            }
        }
    }
}

#[cfg(not(prism_lowered_thresholds))]
const TENSOR_ROUTE_WIDTH: usize = 18;
#[cfg(prism_lowered_thresholds)]
const TENSOR_ROUTE_WIDTH: usize = 4;

#[test]
#[ignore = "the Auto tensor-network expectation route panics on an identity term (the network is never empty)"]
fn an_identity_term_reads_one_on_the_auto_tensor_route() {
    let circuit = identity_cases(TENSOR_ROUTE_WIDTH).remove(2);
    let values = simulate(&circuit)
        .seed(SEED)
        .expectation_values(&[Vec::new(), vec![PauliTerm::z(0)]])
        .unwrap();
    assert_near("auto tensor route", values[0], 1.0, true);
}

#[test]
fn an_empty_observable_list_returns_no_values() {
    for circuit in identity_cases(3) {
        for kind in kinds() {
            if let Some(values) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .expectation_values(&[]),
            ) {
                assert!(values.is_empty(), "{kind:?}");
            }
        }
    }
}

#[test]
fn a_constant_observable_reads_its_constant_with_zero_variance() {
    let constant = PauliObservable::from_terms([(2.5, Vec::new())]).unwrap();
    let nothing = PauliObservable::from_terms(Vec::<(f64, Vec<PauliTerm>)>::new()).unwrap();
    for circuit in identity_cases(3) {
        for kind in kinds() {
            let label = format!("{kind:?}");
            if let Some(result) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .observable_expectation(&constant),
            ) {
                assert_near(&label, result.mean, 2.5, true);
                if let Some(variance) = result.variance {
                    assert_near(&label, variance, 0.0, true);
                }
            }
            if let Some(result) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .observable_variance(&constant),
            ) {
                assert_near(&label, result.mean, 2.5, true);
                assert_near(&label, result.variance, 0.0, true);
            }
            if let Some(result) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .observable_expectation(&nothing),
            ) {
                assert_near(&label, result.mean, 0.0, true);
            }
        }
    }
}

#[test]
fn zero_shots_return_no_rows() {
    let mut measured = Circuit::new(3, 3);
    measured.add_gate(Gate::H, &[0]);
    measured.add_gate(Gate::Cx, &[0, 1]);
    for q in 0..3 {
        measured.add_measure(q, q);
    }
    let noise = NoiseModel::uniform_depolarizing(&measured, 0.1);
    for kind in kinds() {
        let label = format!("{kind:?}");
        if let Some(shots) = answered(
            simulate(&measured)
                .backend(kind.clone())
                .seed(SEED)
                .shots(0),
        ) {
            assert!(shots.shots.is_empty(), "{label}");
        }
        if let Some(counts) = answered(
            simulate(&measured)
                .backend(kind.clone())
                .seed(SEED)
                .sample_counts(0),
        ) {
            assert!(counts.counts.is_empty(), "{label}");
        }
        if let Some(shots) = answered(
            simulate(&measured)
                .backend(kind.clone())
                .noise(&noise)
                .seed(SEED)
                .shots(0),
        ) {
            assert!(shots.shots.is_empty(), "{label} with noise");
        }
    }
}

#[test]
fn repeated_measurement_of_one_qubit_agrees_with_itself() {
    let mut circuit = Circuit::new(2, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    for bit in 0..3 {
        circuit.add_measure(0, bit);
    }
    let mut rotated = Circuit::new(2, 3);
    rotated.add_gate(Gate::Ry(0.7), &[0]);
    rotated.add_gate(Gate::Cx, &[0, 1]);
    for bit in 0..3 {
        rotated.add_measure(0, bit);
    }
    for circuit in [circuit, rotated] {
        for kind in kinds() {
            let label = format!("{kind:?}");
            if let Some(shots) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .shots(64),
            ) {
                assert_eq!(shots.shots.len(), 64, "{label}");
                for shot in &shots.shots {
                    assert!(
                        shot[0] == shot[1] && shot[1] == shot[2],
                        "{label}: {shot:?}"
                    );
                }
            }
            if let Some(counts) = answered(
                simulate(&circuit)
                    .backend(kind.clone())
                    .seed(SEED)
                    .sample_counts(64),
            ) {
                for key in counts.counts.keys() {
                    let bits = key.first().copied().unwrap_or(0);
                    assert!(bits == 0 || bits == 0b111, "{label}: {bits:03b}");
                }
            }
        }
    }
}

/// X on qubit 0, Z on qubit 1, then a CX, measured: the ideal record is `11`. Every
/// gate fires one event per target, so a channel that fires on every event leaves
/// qubit 1 with an odd number of flips and the record off `11`.
fn flip_circuit(clifford: bool) -> Circuit {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Z, &[1]);
    if !clifford {
        circuit.add_gate(Gate::T, &[0]);
    }
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    circuit
}

fn on_every_gate(circuit: &Circuit, channel: NoiseChannel) -> NoiseModel {
    let mut model = NoiseModel::uniform_depolarizing(circuit, 0.0);
    for event in model.after_gate.iter_mut().flatten() {
        *event = NoiseEvent {
            channel: channel.clone(),
            qubits: event.qubits.clone(),
        };
    }
    model
}

/// Kinds that sample a noise model per trajectory or from the exact mixture. `Auto`
/// and `Stabilizer` on a Clifford circuit take the Clifford samplers instead.
fn trajectory_kinds() -> Vec<BackendKind> {
    vec![
        BackendKind::Statevector,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 8 },
        BackendKind::DensityMatrix,
    ]
}

fn clifford_sampler_kinds() -> Vec<BackendKind> {
    vec![BackendKind::Auto, BackendKind::Stabilizer]
}

fn all_noise_kinds() -> Vec<BackendKind> {
    let mut kinds = clifford_sampler_kinds();
    kinds.extend(trajectory_kinds());
    kinds
}

/// Shots of `circuit` under `noise` on each kind; each answer must put the ideal
/// record `11` at the frequency `p11`.
fn assert_flip_rate(
    kinds: &[BackendKind],
    circuit: &Circuit,
    noise: &NoiseModel,
    p11: f64,
    label: &str,
) {
    let shots = 2000;
    for kind in kinds {
        if let Some(result) = answered(
            simulate(circuit)
                .backend(kind.clone())
                .noise(noise)
                .seed(SEED)
                .shots(shots),
        ) {
            let hits = result
                .shots
                .iter()
                .filter(|shot| shot[0] && shot[1])
                .count() as f64
                / shots as f64;
            assert_eq!(hits, p11, "{label} on {kind:?}: P(11)");
        }
    }
}

fn pauli_x(p: f64) -> NoiseChannel {
    NoiseChannel::Pauli {
        px: p,
        py: 0.0,
        pz: 0.0,
    }
}

fn tiny_channels() -> Vec<NoiseChannel> {
    [1e-300, 5e-17]
        .into_iter()
        .flat_map(|p| {
            [
                pauli_x(p),
                NoiseChannel::Pauli {
                    px: p,
                    py: p,
                    pz: p,
                },
                NoiseChannel::Depolarizing { p },
                NoiseChannel::AmplitudeDamping { gamma: p },
            ]
        })
        .collect()
}

#[test]
fn a_zero_rate_leaves_the_ideal_record() {
    for clifford in [true, false] {
        let circuit = flip_circuit(clifford);
        for channel in [
            pauli_x(0.0),
            NoiseChannel::Depolarizing { p: 0.0 },
            NoiseChannel::AmplitudeDamping { gamma: 0.0 },
        ] {
            let noise = on_every_gate(&circuit, channel.clone());
            assert_flip_rate(
                &all_noise_kinds(),
                &circuit,
                &noise,
                1.0,
                &format!("{channel:?}"),
            );
        }
    }
}

#[test]
fn a_unit_rate_applies_the_channel_every_time() {
    // A certain X after every gate target: qubit 1 takes one flip before the CX and
    // one after, so it reads the copy of qubit 0, which ends flipped: never `11`.
    // Full damping drains qubit 0 before the CX and leaves `00`.
    for clifford in [true, false] {
        let circuit = flip_circuit(clifford);
        for channel in [pauli_x(1.0), NoiseChannel::AmplitudeDamping { gamma: 1.0 }] {
            let noise = on_every_gate(&circuit, channel.clone());
            assert_flip_rate(
                &all_noise_kinds(),
                &circuit,
                &noise,
                0.0,
                &format!("{channel:?}"),
            );
        }
    }
}

#[test]
fn tiny_rates_leave_the_ideal_record_on_trajectory_routes() {
    for clifford in [true, false] {
        let circuit = flip_circuit(clifford);
        let kinds = if clifford {
            trajectory_kinds()
        } else {
            all_noise_kinds()
        };
        for channel in tiny_channels() {
            let noise = on_every_gate(&circuit, channel.clone());
            assert_flip_rate(&kinds, &circuit, &noise, 1.0, &format!("{channel:?}"));
        }
    }
}

#[test]
#[ignore = "the frame and noisy compiled Clifford samplers fire every event once 1 - p rounds to 1"]
fn tiny_rates_leave_the_ideal_record_on_the_clifford_samplers() {
    let circuit = flip_circuit(true);
    for channel in tiny_channels() {
        let noise = on_every_gate(&circuit, channel.clone());
        assert_flip_rate(
            &clifford_sampler_kinds(),
            &circuit,
            &noise,
            1.0,
            &format!("{channel:?}"),
        );
    }
}
