//! The Pauli path engine against the density matrix, and its own contracts.
//!
//! Every channel with a Pauli-basis form is checked exactly (`epsilon = 0`,
//! `max_terms = 0`) rather than to a sampling tolerance, because the engine
//! claims exactness there. Truncated runs are checked against the bound they
//! report rather than against a value, since a truncated answer is only ever
//! claimed to sit inside its own discarded mass.

use num_complex::Complex64;
use prism_q::sim::unified_pauli::{run_pauli_path_observable, run_spp_observable};
use prism_q::{
    BackendKind, Circuit, Gate, Instruction, NoiseChannel, NoiseEvent, NoiseModel, PauliTerm,
    circuits, sim,
};

use crate::common::SEED;

const EXACT: f64 = 1e-10;

fn noise_after_every_gate(circuit: &Circuit, channel: NoiseChannel) -> NoiseModel {
    let wants_pair = channel.num_qubits() == 2;
    let after_gate = circuit
        .instructions
        .iter()
        .map(|inst| match inst {
            Instruction::Gate { targets, .. } if wants_pair && targets.len() == 2 => {
                vec![NoiseEvent {
                    channel: channel.clone(),
                    qubits: [targets[0], targets[1]].into_iter().collect(),
                }]
            }
            Instruction::Gate { targets, .. } if !wants_pair => targets
                .iter()
                .map(|&q| NoiseEvent {
                    channel: channel.clone(),
                    qubits: [q].into_iter().collect(),
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    NoiseModel {
        after_gate,
        readout: vec![None; circuit.num_classical_bits],
    }
}

/// Observables spanning all three axes and both weights, so a channel that is
/// wrong on one letter cannot hide behind the others.
fn probe_observables(n: usize) -> Vec<Vec<PauliTerm>> {
    vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1)],
        vec![PauliTerm::y(2)],
        vec![PauliTerm::z(0), PauliTerm::z(n - 1)],
        vec![PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)],
    ]
}

fn pauli_path_values(
    circuit: &Circuit,
    noise: &NoiseModel,
    observables: &[Vec<PauliTerm>],
) -> Vec<f64> {
    sim::simulate(circuit)
        .backend(BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        })
        .noise(noise)
        .seed(SEED)
        .expectation_values(observables)
        .unwrap()
}

fn density_matrix_values(
    circuit: &Circuit,
    noise: &NoiseModel,
    observables: &[Vec<PauliTerm>],
) -> Vec<f64> {
    sim::simulate(circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(noise)
        .seed(SEED)
        .expectation_values(observables)
        .unwrap()
}

#[track_caller]
fn assert_matches_density_matrix(label: &str, n: usize, channel: NoiseChannel) {
    let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
    let noise = noise_after_every_gate(&circuit, channel);
    let observables = probe_observables(n);
    let got = pauli_path_values(&circuit, &noise, &observables);
    let want = density_matrix_values(&circuit, &noise, &observables);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < EXACT,
            "{label}: observable {i} gave {g}, density matrix gave {w}"
        );
    }
}

#[test]
fn pauli_channel_matches_density_matrix() {
    assert_matches_density_matrix(
        "pauli",
        6,
        NoiseChannel::Pauli {
            px: 0.013,
            py: 0.007,
            pz: 0.021,
        },
    );
}

#[test]
fn depolarizing_matches_density_matrix() {
    assert_matches_density_matrix("depolarizing", 6, NoiseChannel::Depolarizing { p: 0.02 });
}

#[test]
fn phase_damping_matches_density_matrix() {
    assert_matches_density_matrix(
        "phase damping",
        6,
        NoiseChannel::PhaseDamping { gamma: 0.05 },
    );
}

// Amplitude damping is the non-unital case: its adjoint sends `Z` to
// `(1 - gamma) Z + gamma I`, so the sum has to grow a term rather than only
// scale one. Dropping that branch leaves every `Z` expectation wrong.
#[test]
fn amplitude_damping_matches_density_matrix() {
    assert_matches_density_matrix(
        "amplitude damping",
        6,
        NoiseChannel::AmplitudeDamping { gamma: 0.08 },
    );
}

#[test]
fn thermal_relaxation_matches_density_matrix() {
    assert_matches_density_matrix(
        "thermal relaxation",
        6,
        NoiseChannel::ThermalRelaxation {
            t1: 40.0,
            t2: 55.0,
            gate_time: 1.5,
            excited_population: 0.0,
        },
    );
}

#[test]
fn hot_thermal_relaxation_matches_density_matrix() {
    assert_matches_density_matrix(
        "hot thermal relaxation",
        6,
        NoiseChannel::ThermalRelaxation {
            t1: 40.0,
            t2: 55.0,
            gate_time: 1.5,
            excited_population: 0.4,
        },
    );
}

#[test]
fn two_qubit_depolarizing_matches_density_matrix() {
    assert_matches_density_matrix(
        "two-qubit depolarizing",
        6,
        NoiseChannel::TwoQubitDepolarizing { p: 0.03 },
    );
}

// The widest exact comparison the host affords, and the one that mixes a
// two-qubit channel with a single-qubit one on the same circuit.
#[test]
fn mixed_channels_match_density_matrix_at_ten_qubits() {
    let n = 10;
    let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
    let mut noise = noise_after_every_gate(&circuit, NoiseChannel::Depolarizing { p: 0.01 });
    for (inst, events) in circuit.instructions.iter().zip(&mut noise.after_gate) {
        if let Instruction::Gate { targets, .. } = inst
            && targets.len() == 2
        {
            events.push(NoiseEvent {
                channel: NoiseChannel::AmplitudeDamping { gamma: 0.02 },
                qubits: [targets[0]].into_iter().collect(),
            });
        }
    }
    let observables = probe_observables(n);
    let got = pauli_path_values(&circuit, &noise, &observables);
    let want = density_matrix_values(&circuit, &noise, &observables);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < EXACT,
            "observable {i} gave {g}, density matrix gave {w}"
        );
    }
}

// A truncated run is only ever claimed to sit inside the coefficient mass it
// dropped, so that is what is asserted. `max_terms` is set below the exact
// run's peak, which is what makes the drop fire at all.
#[test]
fn truncation_error_stays_inside_the_discarded_mass() {
    let n = 8;
    let circuit = circuits::hardware_efficient_ansatz(n, 3, SEED);
    let noise = noise_after_every_gate(&circuit, NoiseChannel::Depolarizing { p: 0.02 });
    let observable = vec![PauliTerm::z(0), PauliTerm::z(n - 1)];

    let exact = run_pauli_path_observable(&circuit, &noise, &observable, 0.0, 0).unwrap();
    assert_eq!(exact.total_discarded, 0.0);

    let truncated =
        run_pauli_path_observable(&circuit, &noise, &observable, 1e-3, exact.peak_terms / 4)
            .unwrap();
    assert!(
        truncated.total_discarded > 0.0,
        "the fixture truncated nothing, so it does not test the bound"
    );
    assert!(
        (truncated.mean - exact.mean).abs() <= truncated.total_discarded + EXACT,
        "error {} exceeded the reported bound {}",
        (truncated.mean - exact.mean).abs(),
        truncated.total_discarded
    );
}

#[test]
fn custom_kraus_is_rejected_naming_the_density_matrix() {
    let circuit = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let kraus = vec![[
        [Complex64::new(0.9, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.9, 0.0)],
    ]];
    let noise = noise_after_every_gate(&circuit, NoiseChannel::Custom { kraus });
    let err = sim::simulate(&circuit)
        .backend(BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        })
        .noise(&noise)
        .seed(SEED)
        .expectation_values(&[vec![PauliTerm::z(0)]])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("DensityMatrix"),
        "rejection should name the route that serves the channel: {err}"
    );
}

#[test]
fn unserved_terminals_are_rejected() {
    let circuit = circuits::hardware_efficient_ansatz(4, 1, SEED);
    let kind = BackendKind::PauliPath {
        epsilon: 0.0,
        max_terms: 0,
    };
    let mut measured = circuit.clone();
    measured.measure_all();
    for err in [
        sim::simulate(&circuit)
            .backend(kind.clone())
            .seed(SEED)
            .run()
            .unwrap_err()
            .to_string(),
        sim::simulate(&circuit)
            .backend(kind.clone())
            .seed(SEED)
            .marginals()
            .unwrap_err()
            .to_string(),
        sim::simulate(&measured)
            .backend(kind.clone())
            .seed(SEED)
            .shots(8)
            .unwrap_err()
            .to_string(),
        sim::simulate(&measured)
            .backend(kind.clone())
            .seed(SEED)
            .sample_counts(8)
            .unwrap_err()
            .to_string(),
    ] {
        assert!(
            err.contains("expectation_values"),
            "rejection should name the terminals it serves: {err}"
        );
    }
}

// With no noise attached the engine is the exact Heisenberg sum, so it has to
// agree with the state vector on the same circuit.
#[test]
fn noiseless_run_matches_the_statevector() {
    let n = 6;
    let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
    let observables = probe_observables(n);
    let got = sim::simulate(&circuit)
        .backend(BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        })
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    let want = sim::simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < EXACT,
            "observable {i} gave {g}, statevector gave {w}"
        );
    }
}

// Observable weight, not circuit width, is what decides whether the sum stays
// small. Both halves are exact counts rather than timings, so they hold on any
// host.
#[test]
fn term_count_is_width_independent_at_unit_weight() {
    let circuit_of = |n| circuits::hardware_efficient_ansatz(n, 2, SEED);
    let mut unit_peak = None;
    let mut unit_discarded = 0.0f64;
    for n in [20usize, 50, 100] {
        let circuit = circuit_of(n);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let got =
            run_pauli_path_observable(&circuit, &noise, &[PauliTerm::z(0)], 1e-3, 1 << 14).unwrap();
        let first = *unit_peak.get_or_insert(got.peak_terms);
        assert_eq!(
            got.peak_terms, first,
            "a weight-1 observable grew from {first} terms at 20 qubits to {} at {n}",
            got.peak_terms
        );
        assert!(
            got.peak_terms < 64,
            "a weight-1 observable held {} terms at {n} qubits",
            got.peak_terms
        );
        unit_discarded = unit_discarded.max(got.total_discarded);
    }

    let n = 50;
    let circuit = circuit_of(n);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let pair = [PauliTerm::z(n / 2), PauliTerm::z(n / 2 + 1)];
    let got = run_pauli_path_observable(&circuit, &noise, &pair, 1e-3, 1 << 14).unwrap();
    assert!(
        got.total_discarded > 100.0 * unit_discarded,
        "weight 2 discarded {} against weight 1's {unit_discarded}, so the guidance that \
         weight decides the regime no longer holds",
        got.total_discarded
    );
}

fn build_embedded_spp_circuit(n: usize, qubits: &[usize; 5]) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[qubits[0]]);
    c.add_gate(Gate::S, &[qubits[1]]);
    c.add_gate(Gate::Cx, &[qubits[0], qubits[2]]);
    c.add_gate(Gate::T, &[qubits[0]]);
    c.add_gate(Gate::Rz(0.37), &[qubits[1]]);
    c.add_gate(Gate::Rzz(-0.41), &[qubits[2], qubits[4]]);
    c.add_gate(Gate::Rx(0.29), &[qubits[3]]);
    c.add_gate(Gate::Ry(-0.19), &[qubits[4]]);
    c.add_pauli_rotation(
        0.53,
        &[
            PauliTerm::x(qubits[0]),
            PauliTerm::y(qubits[2]),
            PauliTerm::z(qubits[4]),
        ],
    );
    c.add_gate(Gate::H, &[qubits[2]]);
    c.add_gate(Gate::H, &[qubits[4]]);
    c
}

#[test]
fn spp_multi_word_observable_and_marginals_match_single_word_embedding() {
    let small_qubits = [0usize, 1, 2, 3, 4];
    let wide_qubits = [0usize, 63, 64, 65, 129];
    let small = build_embedded_spp_circuit(5, &small_qubits);
    let wide = build_embedded_spp_circuit(130, &wide_qubits);

    let obs_small = [PauliTerm::x(1), PauliTerm::y(2), PauliTerm::z(4)];
    let obs_wide = [PauliTerm::x(63), PauliTerm::y(64), PauliTerm::z(129)];

    let res_small = run_spp_observable(&small, &obs_small, 2048, SEED).unwrap();
    let res_wide = run_spp_observable(&wide, &obs_wide, 2048, SEED).unwrap();
    assert_eq!(res_small.mean.to_bits(), res_wide.mean.to_bits());
    assert_eq!(res_small.variance.to_bits(), res_wide.variance.to_bits());
    assert_eq!(
        res_small.nonzero_fraction.to_bits(),
        res_wide.nonzero_fraction.to_bits()
    );

    let marg = sim::simulate(&wide)
        .backend(BackendKind::StochasticPauli { num_samples: 512 })
        .seed(SEED)
        .marginals()
        .unwrap();
    for untouched in [1usize, 62, 66, 128] {
        assert_eq!(marg.marginals[untouched], (1.0, 0.0));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn spp_is_thread_count_deterministic_with_parallel_scratch() {
    let circuit = build_embedded_spp_circuit(70, &[0, 17, 63, 64, 69]);
    let observables = vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(63), PauliTerm::z(64)],
        vec![PauliTerm::y(17), PauliTerm::z(69)],
    ];

    let run = |threads: usize| {
        let pool = prism_q::ThreadPool::with_threads(threads).unwrap();
        pool.install(|| {
            let marg = sim::simulate(&circuit)
                .backend(BackendKind::StochasticPauli { num_samples: 1024 })
                .seed(SEED)
                .marginals()
                .unwrap();
            let exps = sim::simulate(&circuit)
                .backend(BackendKind::StochasticPauli { num_samples: 1024 })
                .seed(SEED)
                .expectation_values(&observables)
                .unwrap();
            (marg.marginals, exps)
        })
    };

    let (m1, e1) = run(1);
    let (m4, e4) = run(4);
    assert_eq!(m1.len(), m4.len());
    for ((p0_a, p1_a), (p0_b, p1_b)) in m1.iter().zip(&m4) {
        assert_eq!(p0_a.to_bits(), p0_b.to_bits());
        assert_eq!(p1_a.to_bits(), p1_b.to_bits());
    }
    for (a, b) in e1.iter().zip(&e4) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
