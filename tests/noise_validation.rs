//! Argument validation on the noise channels and the readout table: which
//! rates a constructor rejects, and which models survive `ensure_pauli_only`.
//! Channel semantics are anchored in `tests/golden_small_circuits.rs`.

use num_complex::Complex64;
use prism_q::circuit::{Circuit, ClassicalCondition, Instruction};
use prism_q::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel};
use prism_q::{
    BackendKind, CircuitBuilder, Gate, PauliTerm, PrismError, density_matrix_expectation_values,
    simulate,
};
use smallvec::smallvec;

fn one_gate_circuit() -> Circuit {
    CircuitBuilder::new_with_classical(1, 1).h(0).build()
}

fn silent_noise(circuit: &Circuit) -> NoiseModel {
    NoiseModel::uniform_depolarizing(circuit, 0.0)
}

fn z0() -> Vec<Vec<PauliTerm>> {
    vec![vec![PauliTerm::z(0)]]
}

#[test]
fn pauli_channel_sum_over_one_rejected() {
    let ch = NoiseChannel::Pauli {
        px: 0.5,
        py: 0.4,
        pz: 0.2,
    };
    assert!(ch.validate().is_err());
}

#[test]
fn pauli_channel_negative_probability_rejected() {
    let ch = NoiseChannel::Pauli {
        px: -0.1,
        py: 0.0,
        pz: 0.0,
    };
    assert!(ch.validate().is_err());
}

#[test]
fn depolarizing_out_of_range_rejected() {
    assert!(NoiseChannel::Depolarizing { p: 1.5 }.validate().is_err());
    assert!(
        NoiseChannel::TwoQubitDepolarizing { p: f64::NAN }
            .validate()
            .is_err()
    );
}

#[test]
fn amplitude_and_phase_damping_validation() {
    assert!(
        NoiseChannel::AmplitudeDamping { gamma: 0.5 }
            .validate()
            .is_ok()
    );
    assert!(
        NoiseChannel::AmplitudeDamping { gamma: -0.1 }
            .validate()
            .is_err()
    );
    assert!(
        NoiseChannel::PhaseDamping {
            gamma: f64::INFINITY
        }
        .validate()
        .is_err()
    );
}

#[test]
fn thermal_relaxation_validation() {
    assert!(
        NoiseChannel::ThermalRelaxation {
            t1: 100.0,
            t2: 50.0,
            gate_time: 1.0,
        }
        .validate()
        .is_ok()
    );
    assert!(
        NoiseChannel::ThermalRelaxation {
            t1: 0.0,
            t2: 1.0,
            gate_time: 1.0,
        }
        .validate()
        .is_err()
    );
    assert!(
        NoiseChannel::ThermalRelaxation {
            t1: 1.0,
            t2: -1.0,
            gate_time: 1.0,
        }
        .validate()
        .is_err()
    );
    assert!(
        NoiseChannel::ThermalRelaxation {
            t1: 1.0,
            t2: 1.0,
            gate_time: -1.0,
        }
        .validate()
        .is_err()
    );
    // t2 must not exceed 2*t1 (outside the amplitude-damping-then-dephasing
    // decomposition's validity).
    assert!(
        NoiseChannel::ThermalRelaxation {
            t1: 50.0,
            t2: 150.0,
            gate_time: 10.0,
        }
        .validate()
        .is_err()
    );
    assert!(
        NoiseChannel::ThermalRelaxation {
            t1: 50.0,
            t2: 100.0,
            gate_time: 10.0,
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn custom_kraus_non_finite_rejected() {
    let bad = NoiseChannel::Custom {
        kraus: vec![[
            [Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]],
    };
    assert!(bad.validate().is_err());

    let bad_im = NoiseChannel::Custom {
        kraus: vec![[
            [Complex64::new(0.0, f64::INFINITY), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]],
    };
    assert!(bad_im.validate().is_err());
}

#[test]
fn custom_kraus_empty_rejected() {
    let empty = NoiseChannel::Custom { kraus: Vec::new() };
    assert!(empty.validate().is_err());
    assert!(!empty.is_exactly_samplable());
}

#[test]
fn ensure_pauli_only_rejects_amplitude_damping() {
    let circuit = one_gate_circuit();
    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.1);
    assert!(noise.ensure_pauli_only().is_err());
    assert!(!noise.is_pauli_only());
    assert!(noise.has_noise());
}

#[test]
fn ensure_pauli_only_rejects_readout() {
    let circuit = one_gate_circuit();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    assert!(noise.ensure_pauli_only().is_ok());
    noise.with_readout_error(0.02, 0.03);
    assert!(noise.ensure_pauli_only().is_err());
    assert!(!noise.is_pauli_only());
}

// A zero-rate entry flips nothing, so it must not cost a model the samplers
// that fold noise in ahead of the draw. This is the reading `reject_readout_at`
// already uses for the mixed-state terminals: inert, not absent.
#[test]
fn zero_rate_readout_stays_pauli_only() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    noise.with_readout_error(0.0, 0.0);
    assert!(noise.is_pauli_only());
    assert!(noise.ensure_pauli_only().is_ok());
    assert!(prism_q::noisy_marginals_analytical(&circuit, &noise, 42).is_ok());
    assert!(prism_q::run_shots_homological(&circuit, &noise, 100, 42).is_ok());

    noise.set_bit_readout_error(0, 0.0, 0.02);
    assert!(!noise.is_pauli_only());
    assert!(prism_q::run_shots_homological(&circuit, &noise, 100, 42).is_err());
}

#[test]
fn readout_p01_out_of_range_rejected() {
    let circuit = one_gate_circuit();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.with_readout_error(1.5, 0.0);
    assert!(noise.validate().is_err());
}

#[test]
fn readout_p10_out_of_range_rejected() {
    let circuit = one_gate_circuit();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.with_readout_error(0.0, f64::NAN);
    assert!(noise.validate().is_err());
}

// `pauli_probs` panics on anything it cannot return as a triple, so the pair
// case gets its own accessor rather than a widened return.
#[test]
fn a_two_qubit_channel_reads_through_the_pair_accessor() {
    let pair = NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.02 },
        qubits: smallvec![3, 1],
    };
    assert_eq!(pair.pauli_pair(), Some((3, 1, 0.02)));
    assert!(pair.channel.as_pauli().is_none());

    let single = NoiseEvent::pauli(0, 0.01, 0.0, 0.0);
    assert!(single.pauli_pair().is_none());
    assert_eq!(single.pauli_probs(), (0.01, 0.0, 0.0));
    assert!(single.channel.pauli_pair_rate().is_none());
}

#[test]
fn validate_rejects_wrong_qubit_count() {
    let circuit = one_gate_circuit();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0].push(NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.01 },
        qubits: smallvec![0],
    });
    assert!(noise.validate().is_err());
}

#[test]
fn validate_rejects_duplicate_two_qubit_targets() {
    let circuit = one_gate_circuit();
    let mut noise = silent_noise(&circuit);
    noise.after_gate[0].push(NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.3 },
        qubits: smallvec![0, 0],
    });
    let err = noise.validate().unwrap_err();
    assert!(
        matches!(&err, PrismError::InvalidParameter { message }
            if message.contains("distinct targets") && message.contains("qubit 0 twice")),
        "{err:?}"
    );
}

// A one-qubit channel on qubit 5 of a two-qubit circuit used to reach the
// density matrix's embedded statevector, where the channel is applied as a
// two-qubit gate on `(q, q + n)` and the write is unchecked.
#[test]
fn out_of_range_one_qubit_channel_rejected_before_evolution() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    let mut noise = silent_noise(&circuit);
    noise.after_gate[0].push(NoiseEvent {
        channel: NoiseChannel::Depolarizing { p: 0.1 },
        qubits: smallvec![5],
    });

    assert!(noise.validate_for(&circuit).is_err());
    let err = density_matrix_expectation_values(&circuit, &z0(), Some(&noise), 42).unwrap_err();
    assert!(
        matches!(&err, PrismError::InvalidParameter { message }
            if message.contains("qubit 5") && message.contains("2-qubit register")),
        "{err:?}"
    );
}

#[test]
fn out_of_range_two_qubit_channel_rejected_on_the_shot_route() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    let mut noise = silent_noise(&circuit);
    noise.after_gate[0].push(NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.1 },
        qubits: smallvec![1, 4],
    });

    let err = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(42)
        .shots(16)
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::InvalidParameter { message } if message.contains("qubit 4")),
        "{err:?}"
    );
}

#[test]
fn noise_model_length_mismatch_rejected() {
    let circuit = one_gate_circuit();
    let mut noise = silent_noise(&circuit);
    noise.after_gate.push(Vec::new());
    assert!(noise.validate_for(&circuit).is_err());
}

// The exact route applies a declared channel literally and the trajectory route
// renormalizes its branch probabilities, so a channel that is not trace
// preserving means two different things depending on where it runs. Rejecting
// it in `validate` is what keeps the two routes agreeing.
#[test]
fn custom_kraus_not_trace_preserving_rejected() {
    let half = Complex64::new(0.5, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let scaled_identity = NoiseChannel::Custom {
        kraus: vec![[[half, zero], [zero, half]]],
    };
    assert!(scaled_identity.validate().is_err());
    assert!(!scaled_identity.is_exactly_samplable());

    let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let bit_flip = NoiseChannel::Custom {
        kraus: vec![
            [[inv_sqrt2, zero], [zero, inv_sqrt2]],
            [[zero, inv_sqrt2], [inv_sqrt2, zero]],
        ],
    };
    assert!(bit_flip.validate().is_ok());
}

#[test]
fn density_matrix_rejects_mid_circuit_measurement() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::H, &[0]);

    let err = density_matrix_expectation_values(&circuit, &z0(), None, 42).unwrap_err();
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "density_matrix"),
        "{err:?}"
    );
}

#[test]
fn density_matrix_rejects_classical_conditional() {
    let mut circuit = Circuit::new(2, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: smallvec![1],
    });

    let err = density_matrix_expectation_values(&circuit, &[vec![PauliTerm::z(1)]], None, 42)
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "density_matrix"),
        "{err:?}"
    );
}

#[test]
fn noise_model_no_noise_when_empty() {
    let circuit = one_gate_circuit();
    let noise = NoiseModel {
        after_gate: vec![Vec::new(); circuit.instructions.len()],
        readout: vec![None; circuit.num_classical_bits],
    };
    assert!(!noise.has_noise());
    assert!(noise.is_pauli_only());
    assert!(noise.validate().is_ok());
}

#[test]
fn noise_event_helpers() {
    let event = NoiseEvent::pauli(3, 0.01, 0.01, 0.01);
    assert_eq!(event.qubit(), 3);
    let (px, py, pz) = event.pauli_probs();
    assert!((px - 0.01).abs() < 1e-15);
    assert!((py - 0.01).abs() < 1e-15);
    assert!((pz - 0.01).abs() < 1e-15);

    let depol = NoiseEvent {
        channel: NoiseChannel::Depolarizing { p: 0.03 },
        qubits: smallvec![0],
    };
    let (px, py, pz) = depol.pauli_probs();
    assert!((px - 0.01).abs() < 1e-15);
    assert!((py - 0.01).abs() < 1e-15);
    assert!((pz - 0.01).abs() < 1e-15);
}
