use num_complex::Complex64;
use prism_q::circuit::Circuit;
use prism_q::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, ReadoutError};
use prism_q::CircuitBuilder;
use smallvec::smallvec;

fn one_gate_circuit() -> Circuit {
    CircuitBuilder::new_with_classical(1, 1).h(0).build()
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
    assert!(NoiseChannel::TwoQubitDepolarizing { p: f64::NAN }
        .validate()
        .is_err());
}

#[test]
fn amplitude_and_phase_damping_validation() {
    assert!(NoiseChannel::AmplitudeDamping { gamma: 0.5 }
        .validate()
        .is_ok());
    assert!(NoiseChannel::AmplitudeDamping { gamma: -0.1 }
        .validate()
        .is_err());
    assert!(NoiseChannel::PhaseDamping {
        gamma: f64::INFINITY
    }
    .validate()
    .is_err());
}

#[test]
fn thermal_relaxation_validation() {
    assert!(NoiseChannel::ThermalRelaxation {
        t1: 100.0,
        t2: 50.0,
        gate_time: 1.0,
    }
    .validate()
    .is_ok());
    assert!(NoiseChannel::ThermalRelaxation {
        t1: 0.0,
        t2: 1.0,
        gate_time: 1.0,
    }
    .validate()
    .is_err());
    assert!(NoiseChannel::ThermalRelaxation {
        t1: 1.0,
        t2: -1.0,
        gate_time: 1.0,
    }
    .validate()
    .is_err());
    assert!(NoiseChannel::ThermalRelaxation {
        t1: 1.0,
        t2: 1.0,
        gate_time: -1.0,
    }
    .validate()
    .is_err());
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

#[test]
fn readout_error_clone_debug() {
    let r = ReadoutError {
        p01: 0.02,
        p10: 0.03,
    };
    let cloned = r.clone();
    assert_eq!(cloned.p01, 0.02);
    assert_eq!(cloned.p10, 0.03);
    let _ = format!("{:?}", r);
}
