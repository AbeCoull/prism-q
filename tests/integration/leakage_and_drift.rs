//! Leakage, quasi-static drift, and schedule-timed idling against their
//! analytic expectations: rates and herald records on the trajectory engines,
//! Gaussian dephasing and correlated draws from the drift rules, and T1/T2
//! decay over a known schedule.

use prism_q::circuit::Circuit;
use prism_q::{
    BackendKind, CircuitBuilder, DeviceCalibration, DriftDistribution, Gate, GateCalibration,
    GateFilter, GateTimes, NoiseBuilder, NoiseChannel, NoiseEvent, NoiseModel, PrismError,
    QubitCalibration, ShotsResult, simulate,
};
use smallvec::smallvec;

const SEED: u64 = 42;

fn empty_model(circuit: &Circuit) -> NoiseModel {
    NoiseModel {
        after_gate: vec![Vec::new(); circuit.instructions.len()],
        readout: vec![None; circuit.num_classical_bits],
    }
}

fn event(channel: NoiseChannel, qubits: &[usize]) -> NoiseEvent {
    NoiseEvent {
        channel,
        qubits: qubits.iter().copied().collect(),
    }
}

fn shots(circuit: &Circuit, noise: &NoiseModel, kind: BackendKind, n: usize) -> ShotsResult {
    simulate(circuit)
        .backend(kind)
        .noise(noise)
        .seed(SEED)
        .shots(n)
        .unwrap()
}

fn rate(result: &ShotsResult, bit: usize) -> f64 {
    result.shots.iter().filter(|shot| shot[bit]).count() as f64 / result.shots.len() as f64
}

fn herald_rate(result: &ShotsResult, qubit: usize) -> f64 {
    let leaked = result
        .leaked
        .as_ref()
        .expect("leakage model carries a herald record");
    leaked.iter().filter(|shot| shot[qubit]).count() as f64 / leaked.len() as f64
}

/// Five binomial standard errors around `p` over `n` shots, floored for rates near 0 or 1.
fn assert_rate(observed: f64, p: f64, n: usize, label: &str) {
    let tolerance = 5.0 * (p * (1.0 - p) / n as f64).sqrt().max(1e-3);
    assert!(
        (observed - p).abs() <= tolerance,
        "{label}: observed {observed}, expected {p} +- {tolerance}"
    );
}

/// `x` on qubit 0, a two-qubit gate pairing it with qubit 1, then `extra` on 1.
fn pair_circuit(prepare_plus: bool) -> Circuit {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::X, &[0]);
    if prepare_plus {
        circuit.add_gate(Gate::H, &[1]);
    }
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::X, &[0]);
    if prepare_plus {
        circuit.add_gate(Gate::H, &[1]);
    }
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    circuit
}

#[test]
fn leaked_qubit_reports_one_blocks_gates_and_depolarizes_its_partner() {
    for prepare_plus in [false, true] {
        let circuit = pair_circuit(prepare_plus);
        let mut noise = empty_model(&circuit);
        noise.after_gate[0].push(event(NoiseChannel::Leakage { p: 1.0 }, &[0]));
        let n = 20_000;
        for kind in [
            BackendKind::Statevector,
            BackendKind::Mps { max_bond_dim: 16 },
        ] {
            let result = shots(&circuit, &noise, kind.clone(), n);
            assert_eq!(rate(&result, 0), 1.0, "a leaked qubit reads 1 on {kind:?}");
            assert_eq!(herald_rate(&result, 0), 1.0);
            assert_eq!(herald_rate(&result, 1), 0.0);
            // A uniform Pauli flips a Z outcome on X and Y and an X outcome on Y
            // and Z: one half either way.
            assert_rate(rate(&result, 1), 0.5, n, "partner depolarization");
        }
    }
}

#[test]
fn transport_spreads_leakage_across_the_pair_at_its_rate() {
    let circuit = pair_circuit(false);
    let mut noise = empty_model(&circuit);
    noise.after_gate[0].push(event(NoiseChannel::Leakage { p: 1.0 }, &[0]));
    let p_transport = 0.3;
    noise.after_gate[1].push(event(
        NoiseChannel::LeakageTransport { p: p_transport },
        &[0, 1],
    ));
    let n = 20_000;
    let result = shots(&circuit, &noise, BackendKind::Statevector, n);
    assert_rate(herald_rate(&result, 1), p_transport, n, "transport herald");
    let leaked = result.leaked.as_ref().unwrap();
    for (shot, flags) in result.shots.iter().zip(leaked) {
        if flags[1] {
            assert!(shot[1], "a qubit leaked by transport reads 1");
        }
    }
    assert_rate(
        rate(&result, 1),
        p_transport + (1.0 - p_transport) * 0.5,
        n,
        "partner outcome",
    );
}

#[test]
fn reset_clears_the_leak_flag_and_keeps_the_herald() {
    let mut circuit = Circuit::new(1, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_reset(0);
    circuit.add_measure(0, 1);
    let mut noise = empty_model(&circuit);
    noise.after_gate[0].push(event(NoiseChannel::Leakage { p: 1.0 }, &[0]));
    let result = shots(&circuit, &noise, BackendKind::Statevector, 64);
    for (shot, flags) in result.shots.iter().zip(result.leaked.as_ref().unwrap()) {
        assert_eq!(shot, &vec![true, false]);
        assert_eq!(flags, &vec![true]);
    }
}

/// Probability a qubit reads 1 after `steps` slots of leakage `p_leak` then
/// seepage `p_seep`, from the chain over never leaked (reads 0), leaked (reads 1)
/// and returned (reads 1 half the time).
fn leak_seep_chain(steps: usize, p_leak: f64, p_seep: f64) -> (f64, f64) {
    let (mut never, mut leaked, mut returned) = (1.0, 0.0, 0.0);
    for _ in 0..steps {
        let fresh = never * p_leak;
        never -= fresh;
        let relapse = returned * p_leak;
        returned -= relapse;
        leaked += fresh + relapse;
        let seep = leaked * p_seep;
        leaked -= seep;
        returned += seep;
    }
    (leaked + 0.5 * returned, 1.0 - never)
}

#[test]
fn leaked_population_reaches_the_leak_seep_steady_state() {
    let (p_leak, p_seep, steps) = (0.05, 0.1, 200);
    let mut circuit = Circuit::new(1, 1);
    for _ in 0..steps {
        circuit.add_gate(Gate::Id, &[0]);
    }
    circuit.add_measure(0, 0);
    let noise = NoiseBuilder::new()
        .after_gates(GateFilter::all(), NoiseChannel::Leakage { p: p_leak })
        .after_gates(GateFilter::all(), NoiseChannel::Seepage { p: p_seep })
        .build(&circuit)
        .unwrap();

    let n = 20_000;
    let result = shots(&circuit, &noise, BackendKind::Statevector, n);
    let (expected_one, expected_herald) = leak_seep_chain(steps, p_leak, p_seep);
    assert_rate(rate(&result, 0), expected_one, n, "reads 1");
    assert_rate(herald_rate(&result, 0), expected_herald, n, "herald");

    let steady = p_leak * (1.0 - p_seep) / (1.0 - (1.0 - p_leak) * (1.0 - p_seep));
    let leaked = 2.0 * rate(&result, 0) - 1.0;
    assert!(
        (leaked - steady).abs() < 0.03,
        "steady leaked population {leaked} against {steady}"
    );
}

#[test]
fn leakage_and_drift_decline_the_density_matrix() {
    let circuit = CircuitBuilder::new_with_classical(1, 1)
        .h(0)
        .measure_all()
        .build();
    let mut noise = empty_model(&circuit);
    noise.after_gate[0].push(event(NoiseChannel::Leakage { p: 0.1 }, &[0]));
    let err = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .shots(8)
        .unwrap_err();
    assert!(
        matches!(err, PrismError::IncompatibleBackend { .. }),
        "{err}"
    );

    let drift = NoiseBuilder::new()
        .over_rotation_drift(GateFilter::all(), 0.1)
        .build(
            &CircuitBuilder::new_with_classical(1, 1)
                .rx(0.5, 0)
                .measure_all()
                .build(),
        )
        .unwrap();
    assert!(
        drift.after_gate[0]
            .iter()
            .any(|e| matches!(e.channel, NoiseChannel::QuasiStatic { .. }))
    );
}

/// `h`, a delay of `delay` seconds through a timed identity, `h`, measure.
fn ramsey(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits, num_qubits);
    for q in 0..num_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..num_qubits {
        circuit.add_gate(Gate::Id, &[q]);
    }
    for q in 0..num_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..num_qubits {
        circuit.add_measure(q, q);
    }
    circuit
}

#[test]
fn detuning_ramsey_decays_with_the_gaussian_envelope() {
    let sigma = 2.0e5;
    let n = 20_000;
    for delay in [2.0e-6, 5.0e-6, 1.0e-5] {
        let circuit = ramsey(1);
        let noise = NoiseBuilder::new()
            .schedule(GateTimes::new(0.0, 0.0).with_gate("id", delay))
            .quasi_static_detuning(DriftDistribution::independent([sigma]))
            .build(&circuit)
            .unwrap();
        let result = shots(&circuit, &noise, BackendKind::Statevector, n);
        let envelope = (-(sigma * delay).powi(2) / 2.0).exp();
        assert_rate(rate(&result, 0), (1.0 - envelope) / 2.0, n, "ramsey");
    }
}

#[test]
fn correlated_detuning_moves_two_ramsey_fringes_together() {
    let sigma = 1.0e5;
    let delay = 1.0e-5;
    let circuit = ramsey(2);
    let n = 20_000;
    let both = |rho: f64| {
        let drift = DriftDistribution::independent([sigma, sigma])
            .with_neighbour_correlation([(0, 1)], rho);
        let noise = NoiseBuilder::new()
            .schedule(GateTimes::new(0.0, 0.0).with_gate("id", delay))
            .quasi_static_detuning(drift)
            .build(&circuit)
            .unwrap();
        let result = shots(&circuit, &noise, BackendKind::Statevector, n);
        result.shots.iter().filter(|s| s[0] && s[1]).count() as f64 / n as f64
    };
    // With x = sigma * delay = 1, P(1) = (1 - cos(x z)) / 2 per qubit.
    let e1 = (-0.5f64).exp();
    let independent = ((1.0 - e1) / 2.0).powi(2);
    let locked = (1.0 - 2.0 * e1 + (1.0 + (-2.0f64).exp()) / 2.0) / 4.0;
    assert_rate(both(0.0), independent, n, "independent");
    assert_rate(both(1.0), locked, n, "fully correlated");
}

#[test]
fn shared_amplitude_drift_over_rotates_a_family_coherently() {
    let sigma = 0.2;
    let n = 20_000;
    let circuit = CircuitBuilder::new_with_classical(2, 2)
        .rx(std::f64::consts::PI, 0)
        .rx(std::f64::consts::PI, 1)
        .measure_all()
        .build();
    let noise = NoiseBuilder::new()
        .over_rotation_drift(GateFilter::all().named("rx"), sigma)
        .build(&circuit)
        .unwrap();
    let result = shots(&circuit, &noise, BackendKind::Statevector, n);
    // Rx(pi (1 + e)) leaves |0> with probability sin^2(pi e / 2).
    let x = std::f64::consts::PI * sigma;
    let p0 = (1.0 - (-x * x / 2.0).exp()) / 2.0;
    assert_rate(1.0 - rate(&result, 0), p0, n, "qubit 0 stays");
    // One draw per shot drives both gates, so the two misses coincide.
    let both = result.shots.iter().filter(|s| !s[0] && !s[1]).count() as f64 / n as f64;
    let locked = (3.0 - 4.0 * (-x * x / 2.0).exp() + (-2.0 * x * x).exp()) / 8.0;
    assert_rate(both, locked, n, "shared draw");
}

#[test]
fn drift_rules_report_bad_inputs() {
    let circuit = ramsey(2);
    let no_schedule = NoiseBuilder::new()
        .quasi_static_detuning(DriftDistribution::independent([1.0, 1.0]))
        .build(&circuit);
    assert!(no_schedule.is_err());

    let too_narrow = NoiseBuilder::new()
        .schedule(GateTimes::new(1e-8, 1e-7))
        .quasi_static_detuning(DriftDistribution::independent([1.0]))
        .build(&circuit);
    assert!(too_narrow.is_err());

    let not_psd = DriftDistribution::from_covariance(vec![vec![1.0, 2.0], vec![2.0, 1.0]]);
    let rejected = NoiseBuilder::new()
        .schedule(GateTimes::new(1e-8, 1e-7))
        .quasi_static_detuning(not_psd)
        .build(&circuit);
    assert!(rejected.is_err());

    let both_idle = NoiseBuilder::new()
        .schedule(GateTimes::new(1e-8, 1e-7))
        .on_idle_qubits(NoiseChannel::PhaseDamping { gamma: 0.01 })
        .scheduled_idle([(1e-4, 1e-4), (1e-4, 1e-4)])
        .build(&circuit);
    assert!(both_idle.is_err());

    let negative = NoiseBuilder::new()
        .schedule(GateTimes::new(-1.0, 1e-7))
        .scheduled_idle([(1e-4, 1e-4), (1e-4, 1e-4)])
        .build(&circuit);
    assert!(negative.is_err());
}

fn exact_p1(circuit: &Circuit, noise: &NoiseModel, qubit: usize) -> f64 {
    simulate(circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(noise)
        .seed(SEED)
        .marginals()
        .unwrap()
        .marginals[qubit]
        .1
}

#[test]
fn scheduled_idling_follows_t1_and_t2_over_the_schedule() {
    let (t1, t2) = (50e-6, 30e-6);
    let tau = 1e-6;
    let layers = 8;
    // Qubit 0 is busy in the first layer and idle for the other `layers - 1`,
    // until the barrier lets its last gate and measurement run.
    let build = |hadamard: bool| {
        let mut circuit = Circuit::new(2, 1);
        circuit.add_gate(if hadamard { Gate::H } else { Gate::X }, &[0]);
        for _ in 0..layers {
            circuit.add_gate(Gate::X, &[1]);
        }
        circuit.add_barrier(&[0, 1]);
        if hadamard {
            circuit.add_gate(Gate::H, &[0]);
        }
        circuit.add_measure(0, 0);
        circuit
    };
    let noise_for = |circuit: &Circuit| {
        NoiseBuilder::new()
            .schedule(GateTimes::new(tau, 4.0 * tau))
            .scheduled_idle([(t1, t2), (t1, t2)])
            .build(circuit)
            .unwrap()
    };
    let idle = (layers - 1) as f64 * tau;

    let relax = build(false);
    let p1 = exact_p1(&relax, &noise_for(&relax), 0);
    assert!((p1 - (-idle / t1).exp()).abs() < 1e-12, "T1: {p1}");

    // The closing `h` shares its layer with nothing, so qubit 1 idles there
    // while qubit 0 does not; qubit 0 dephases over the same `idle`.
    let ramsey = build(true);
    let p0 = 1.0 - exact_p1(&ramsey, &noise_for(&ramsey), 0);
    assert!(
        (p0 - (1.0 + (-idle / t2).exp()) / 2.0).abs() < 1e-12,
        "T2: {p0}"
    );

    let n = 20_000;
    let sampled = shots(&relax, &noise_for(&relax), BackendKind::Statevector, n);
    assert_rate(rate(&sampled, 0), (-idle / t1).exp(), n, "trajectory T1");
}

#[test]
fn a_shorter_gate_idles_for_the_rest_of_its_layer() {
    let (t1, t2) = (20e-6, 20e-6);
    let (one, two) = (50e-9, 2e-6);
    let mut circuit = Circuit::new(3, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_measure(0, 0);
    let noise = NoiseBuilder::new()
        .schedule(GateTimes::new(one, two))
        .scheduled_idle([(t1, t2); 3])
        .build(&circuit)
        .unwrap();
    let p1 = exact_p1(&circuit, &noise, 0);
    assert!((p1 - (-(two - one) / t1).exp()).abs() < 1e-12, "{p1}");

    // Listed after the measurement, the `cx` still shares the first layer, so
    // qubit 0 decays over its remainder before it is read.
    let mut reordered = Circuit::new(3, 1);
    reordered.add_gate(Gate::X, &[0]);
    reordered.add_measure(0, 0);
    reordered.add_gate(Gate::Cx, &[1, 2]);
    let noise = NoiseBuilder::new()
        .schedule(GateTimes::new(one, two))
        .scheduled_idle([(t1, t2); 3])
        .build(&reordered)
        .unwrap();
    assert!(noise.after_gate[0].iter().any(|event| event.qubits[0] == 0
        && matches!(event.channel, NoiseChannel::ThermalRelaxation { gate_time, .. }
            if (gate_time - (two - one)).abs() < 1e-18)));
    let n = 20_000;
    let sampled = shots(&reordered, &noise, BackendKind::Statevector, n);
    assert_rate(rate(&sampled, 0), (-(two - one) / t1).exp(), n, "reordered");
}

#[test]
fn calibration_schedule_adds_idle_relaxation_to_the_gate_noise() {
    let qubit = QubitCalibration {
        t1: 80e-6,
        t2: 60e-6,
        p01: 0.0,
        p10: 0.0,
    };
    let calibration = DeviceCalibration::new(
        vec![qubit; 2],
        GateCalibration {
            time: 40e-9,
            error: 0.0,
        },
        GateCalibration {
            time: 400e-9,
            error: 0.0,
        },
    )
    .unwrap();
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Cx, &[1, 0]);
    let wide = calibration.to_scheduled_noise_model(&circuit);
    assert!(wide.is_err(), "a circuit wider than the table is rejected");

    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::Cx, &[1, 0]);
    let plain = calibration.to_noise_model(&circuit).unwrap();
    let scheduled = calibration.to_scheduled_noise_model(&circuit).unwrap();
    assert_eq!(plain.after_gate[0], scheduled.after_gate[0]);
    let added = scheduled.after_gate[1].len() - plain.after_gate[1].len();
    assert_eq!(added, 0, "both qubits are busy for the whole first layer");
    assert_eq!(scheduled.after_gate[2].len(), plain.after_gate[2].len());
}

#[test]
fn models_without_memory_record_no_leak_herald() {
    let circuit = CircuitBuilder::new_with_classical(2, 2)
        .h(0)
        .cx(0, 1)
        .measure_all()
        .build();
    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.05);
    assert!(
        shots(&circuit, &noise, BackendKind::Statevector, 16)
            .leaked
            .is_none()
    );
    let mut drift = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    drift.after_gate[0].push(NoiseEvent {
        channel: NoiseChannel::QuasiStatic {
            axis: prism_q::PauliAxis::X,
            weights: vec![(0, 0.1)],
        },
        qubits: smallvec![0],
    });
    assert!(
        shots(&circuit, &drift, BackendKind::Statevector, 16)
            .leaked
            .is_none()
    );
}
