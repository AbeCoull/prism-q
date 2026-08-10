use prism_q::circuit::SmallVec;
use prism_q::*;

fn run_shots_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    simulate(circuit).backend(kind).seed(seed).shots(num_shots)
}

fn run_shots_with_noise(
    kind: BackendKind,
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    simulate(circuit)
        .backend(kind)
        .noise(noise)
        .seed(seed)
        .shots(num_shots)
}

#[test]
fn depolarizing_trajectory_cross_check_with_compiled() {
    let n = 6;
    let mut circuit = Circuit::new(n, n);
    circuit.add_gate(Gate::H, &[0]);
    for i in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let num_shots = 2000;

    let compiled =
        run_shots_with_noise(BackendKind::Stabilizer, &circuit, &noise, num_shots, 42).unwrap();
    let trajectory =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, num_shots, 42).unwrap();

    let all_zero: Vec<bool> = vec![false; n];
    let all_one: Vec<bool> = vec![true; n];

    let c_00 = compiled.shots.iter().filter(|s| **s == all_zero).count();
    let c_11 = compiled.shots.iter().filter(|s| **s == all_one).count();
    let t_00 = trajectory.shots.iter().filter(|s| **s == all_zero).count();
    let t_11 = trajectory.shots.iter().filter(|s| **s == all_one).count();

    assert!(c_00 > 500, "compiled: too few |0...0⟩: {c_00}");
    assert!(c_11 > 500, "compiled: too few |1...1⟩: {c_11}");
    assert!(t_00 > 500, "trajectory: too few |0...0⟩: {t_00}");
    assert!(t_11 > 500, "trajectory: too few |1...1⟩: {t_11}");

    let c_other = num_shots - c_00 - c_11;
    let t_other = num_shots - t_00 - t_11;
    assert!(c_other > 0, "compiled should produce noise errors");
    assert!(t_other > 0, "trajectory should produce noise errors");
}

#[test]
fn amplitude_damping_analytic_single_qubit() {
    let gamma = 0.8;
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);

    let noise = NoiseModel::with_amplitude_damping(&circuit, gamma);
    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 5000, 42).unwrap();

    let num_zero = result.shots.iter().filter(|s| !s[0]).count();
    let p_zero = num_zero as f64 / 5000.0;
    assert!(
        (p_zero - gamma).abs() < 0.05,
        "P(0) should be ~gamma={gamma}, got {p_zero}"
    );
}

// Each damping event reads the target's reduced density matrix, which the
// tensor network answers by contracting against its conjugate with the other
// qubit traced out.
#[test]
fn tensor_network_amplitude_damping_analytic() {
    let gamma = 0.8;
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);

    let noise = NoiseModel::with_amplitude_damping(&circuit, gamma);
    let result =
        run_shots_with_noise(BackendKind::TensorNetwork, &circuit, &noise, 5000, 42).unwrap();

    for q in 0..2 {
        let p_zero = result.shots.iter().filter(|s| !s[q]).count() as f64 / 5000.0;
        assert!(
            (p_zero - gamma).abs() < 0.05,
            "q{q}: P(0) should be ~gamma={gamma}, got {p_zero}"
        );
    }
}

#[test]
fn amplitude_damping_ground_state_unchanged() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_measure(0, 0);

    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.9);
    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 1000, 42).unwrap();

    let num_one = result.shots.iter().filter(|s| s[0]).count();
    assert_eq!(num_one, 0, "AD on |0⟩ should never produce |1⟩");
}

#[test]
fn phase_damping_no_population_change() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    for events in &mut noise.after_gate {
        *events = events
            .iter()
            .map(|e| NoiseEvent {
                channel: NoiseChannel::PhaseDamping { gamma: 0.95 },
                qubits: e.qubits.clone(),
            })
            .collect();
    }

    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 1000, 42).unwrap();

    let num_one = result.shots.iter().filter(|s| s[0]).count();
    assert_eq!(num_one, 1000, "PD should not flip |1⟩ to |0⟩");
}

#[test]
fn readout_error_through_public_api() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.with_readout_error(0.2, 0.0);

    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 5000, 42).unwrap();

    let num_one = result.shots.iter().filter(|s| s[0]).count();
    let p_one = num_one as f64 / 5000.0;
    assert!(
        (p_one - 0.2).abs() < 0.05,
        "readout p01=0.2 should flip ~20% of |0⟩, got {p_one}"
    );
}

#[test]
fn two_qubit_depolarizing_through_public_api() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[1] = vec![NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.3 },
        qubits: SmallVec::from_slice(&[0, 1]),
    }];

    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 2000, 42).unwrap();

    let bell_00 = result.shots.iter().filter(|s| !s[0] && !s[1]).count();
    let bell_11 = result.shots.iter().filter(|s| s[0] && s[1]).count();
    let other = 2000 - bell_00 - bell_11;

    assert!(
        other > 100,
        "2q depolarizing p=0.3 should produce non-Bell outcomes, got {other} errors"
    );
}

#[test]
fn zero_noise_deterministic_across_backends() {
    let n = 4;
    let mut circuit = Circuit::new(n, n);
    circuit.add_gate(Gate::H, &[0]);
    for i in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);

    let r1 = run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 100, 42).unwrap();
    let r2 = run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 100, 42).unwrap();

    assert_eq!(r1.shots, r2.shots, "same seed must produce same results");

    for shot in &r1.shots {
        let all_same = shot.iter().all(|&b| b == shot[0]);
        assert!(
            all_same,
            "zero noise GHZ must be all-0 or all-1: {:?}",
            shot
        );
    }
}

#[test]
fn thermal_relaxation_preserves_superposition_statistics() {
    // With a very short gate_time (p_reset ≈ 0, p_dephase ≈ 0), thermal
    // relaxation is nearly identity and the measurement distribution stays
    // ~50/50. Guards against the old X-then-project bug which would
    // incorrectly collapse |+⟩.
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0] = vec![NoiseEvent {
        channel: NoiseChannel::ThermalRelaxation {
            t1: 1.0,
            t2: 1.0,
            gate_time: 1e-6,
        },
        qubits: SmallVec::from_slice(&[0]),
    }];

    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 2000, 42).unwrap();
    let num_one = result.shots.iter().filter(|s| s[0]).count();
    let p_one = num_one as f64 / 2000.0;
    assert!(
        (p_one - 0.5).abs() < 0.05,
        "|+⟩ with near-zero thermal relaxation should measure ~50/50, got {p_one}"
    );
}

#[test]
fn thermal_relaxation_strong_reset_to_ground() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0] = vec![NoiseEvent {
        channel: NoiseChannel::ThermalRelaxation {
            t1: 1.0,
            t2: 1.0,
            gate_time: 2.3,
        },
        qubits: SmallVec::from_slice(&[0]),
    }];

    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 2000, 42).unwrap();
    let num_zero = result.shots.iter().filter(|s| !s[0]).count();
    let p_zero = num_zero as f64 / 2000.0;
    // p_reset = 1 - exp(-2.3) ≈ 0.9
    assert!(
        p_zero > 0.80,
        "strong thermal relaxation on |1⟩ should decay to |0⟩ ~90%, got {p_zero}"
    );
}

#[test]
fn reset_after_superposition() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    let result = run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 200, 42).unwrap();
    let num_one = result.shots.iter().filter(|s| s[0]).count();
    assert_eq!(num_one, 0, "reset after H should always measure 0");
}

#[test]
fn reset_from_excited_state_stabilizer() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);

    let result = run_shots_with(BackendKind::Stabilizer, &circuit, 100, 42).unwrap();
    let num_one = result.shots.iter().filter(|s| s[0]).count();
    assert_eq!(num_one, 0, "reset must clear |1⟩ to |0⟩");
}

#[test]
fn stabilizer_rejects_amplitude_damping() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);

    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.5);
    let err = run_shots_with_noise(BackendKind::Stabilizer, &circuit, &noise, 10, 42);
    assert!(
        err.is_err(),
        "stabilizer backend must reject non-Pauli noise, got {err:?}"
    );
}

#[test]
fn openqasm_reset_statement_parses() {
    let qasm = "\
        OPENQASM 3.0;\n\
        qubit[2] q;\n\
        bit[2] c;\n\
        x q[0];\n\
        reset q[0];\n\
        c[0] = measure q[0];\n\
        c[1] = measure q[1];\n\
    ";
    let circuit = prism_q::circuit::openqasm::parse(qasm).expect("reset should parse");
    assert!(circuit.has_resets(), "circuit should contain a reset");

    let result = run_shots_with(BackendKind::Statevector, &circuit, 50, 42).unwrap();
    for shot in &result.shots {
        assert!(
            !shot[0],
            "reset of q[0] should always measure 0, got {:?}",
            shot
        );
    }
}

#[test]
fn openqasm_reset_broadcast_parses() {
    let qasm = "\
        OPENQASM 3.0;\n\
        qubit[3] q;\n\
        bit[3] c;\n\
        x q[0];\n\
        x q[1];\n\
        x q[2];\n\
        reset q;\n\
        c[0] = measure q[0];\n\
        c[1] = measure q[1];\n\
        c[2] = measure q[2];\n\
    ";
    let circuit = prism_q::circuit::openqasm::parse(qasm).expect("broadcast reset should parse");
    let result = run_shots_with(BackendKind::Statevector, &circuit, 50, 42).unwrap();
    for shot in &result.shots {
        assert!(
            !shot[0] && !shot[1] && !shot[2],
            "broadcast reset should zero all qubits, got {:?}",
            shot
        );
    }
}

#[test]
fn custom_kraus_bit_flip_channel() {
    use num_complex::Complex64;

    let gamma: f64 = 0.4;
    let sg = gamma.sqrt();
    let s0 = (1.0 - gamma).sqrt();
    let zero = Complex64::new(0.0, 0.0);

    let k0 = [
        [Complex64::new(s0, 0.0), zero],
        [zero, Complex64::new(s0, 0.0)],
    ];
    let k1 = [
        [zero, Complex64::new(sg, 0.0)],
        [Complex64::new(sg, 0.0), zero],
    ];

    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::Id, &[0]);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0] = vec![NoiseEvent {
        channel: NoiseChannel::Custom {
            kraus: vec![k0, k1],
        },
        qubits: SmallVec::from_slice(&[0]),
    }];

    let result =
        run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 5000, 42).unwrap();

    let num_one = result.shots.iter().filter(|s| s[0]).count();
    let p_one = num_one as f64 / 5000.0;
    assert!(
        (p_one - gamma).abs() < 0.05,
        "custom bit-flip kraus gamma={gamma} should flip ~40%, got {p_one}"
    );
}

#[test]
fn custom_kraus_dense_channel_uses_coherence() {
    use num_complex::Complex64;

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let zero = Complex64::new(0.0, 0.0);
    let plus = Complex64::new(inv_sqrt2, 0.0);
    let minus = Complex64::new(-inv_sqrt2, 0.0);

    let k0 = [[plus, plus], [zero, zero]];
    let k1 = [[zero, zero], [plus, minus]];

    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0] = vec![NoiseEvent {
        channel: NoiseChannel::Custom {
            kraus: vec![k0, k1],
        },
        qubits: SmallVec::from_slice(&[0]),
    }];

    let backends = vec![
        BackendKind::Statevector,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 64 },
        BackendKind::ProductState,
        BackendKind::Factored,
    ];

    for backend in backends {
        let result = run_shots_with_noise(backend.clone(), &circuit, &noise, 200, 42).unwrap();

        let num_one = result.shots.iter().filter(|s| s[0]).count();
        assert_eq!(
            num_one, 0,
            "dense custom Kraus should project |+> through the K0 branch deterministically on {backend:?}"
        );
    }
}

// `H|0>` damped at `gamma` has exact `P(1) = (1 - gamma) / 2`. With the noise
// sampler and the backend sharing a ChaCha stream, the jump fired on exactly the
// shots that would have measured `|0>` anyway and this read 0.384 against 0.425,
// 37 standard errors out. The tight bound is what holds the two streams apart.
#[test]
fn noise_and_measurement_draws_are_independent() {
    let gamma = 0.15;
    let num_shots = 100_000;
    let exact = (1.0 - gamma) / 2.0;

    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    let noise = NoiseModel::with_amplitude_damping(&circuit, gamma);

    for kind in [BackendKind::Statevector, BackendKind::Factored] {
        let result = run_shots_with_noise(kind.clone(), &circuit, &noise, num_shots, 42).unwrap();
        let ones = result.shots.iter().filter(|shot| shot[0]).count();
        let p_one = ones as f64 / num_shots as f64;
        let sigma = (exact * (1.0 - exact) / num_shots as f64).sqrt();

        assert!(
            (p_one - exact).abs() <= 5.0 * sigma,
            "{kind:?}: P(1) = {p_one}, exact = {exact}, off by {} standard errors",
            (p_one - exact).abs() / sigma
        );
    }
}

/// Two independent two-qubit blocks with no gate bridging them, so the factored
/// backend holds a product of sub-states rather than one dense block.
fn damped_two_block_circuit(num_classical: usize) -> Circuit {
    let mut circuit = Circuit::new(4, num_classical);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::Ry(0.7), &[3]);
    circuit
}

// The non-Pauli trajectory path on the factored backend: the engine reads
// `qubit_probability`, samples a branch, and applies that branch's Kraus operator
// through `Backend::apply_1q_matrix`. The density matrix evolves the same channel
// exactly with no sampling, so it is the reference here, not a second opinion.
#[test]
fn factored_amplitude_damping_matches_density_matrix() {
    let gamma = 0.15;
    let num_shots = 8000;

    let unitary = damped_two_block_circuit(0);
    let mut measured = damped_two_block_circuit(4);
    for q in 0..4 {
        measured.add_measure(q, q);
    }

    // `with_amplitude_damping` attaches events only after gate instructions, so
    // the two models agree on every index the gates occupy.
    let reference_noise = NoiseModel::with_amplitude_damping(&unitary, gamma);
    let observables: Vec<Vec<PauliTerm>> = (0..4).map(|q| vec![PauliTerm::z(q)]).collect();
    let exact =
        density_matrix_expectation_values(&unitary, &observables, Some(&reference_noise), 42)
            .unwrap();

    let noise = NoiseModel::with_amplitude_damping(&measured, gamma);
    let result =
        run_shots_with_noise(BackendKind::Factored, &measured, &noise, num_shots, 42).unwrap();

    for q in 0..4 {
        let ones = result.shots.iter().filter(|shot| shot[q]).count();
        let p_one = ones as f64 / num_shots as f64;
        let measured_z = 1.0 - 2.0 * p_one;

        let p_exact = (1.0 - exact[q]) / 2.0;
        // Standard error of <Z> from `num_shots` Bernoulli draws, floored so a
        // near-deterministic qubit still admits the last few bits of rounding.
        let sigma = (2.0 * (p_exact * (1.0 - p_exact) / num_shots as f64).sqrt()).max(1e-9);

        assert!(
            (measured_z - exact[q]).abs() <= 5.0 * sigma,
            "factored trajectory <Z_{q}> = {measured_z}, density matrix = {}, \
             difference {} exceeds 5 sigma ({})",
            exact[q],
            (measured_z - exact[q]).abs(),
            5.0 * sigma
        );
    }
}

// The same channel through the density matrix and the statevector, so a
// disagreement above names the factored backend rather than the reference.
#[test]
fn statevector_amplitude_damping_matches_density_matrix() {
    let gamma = 0.15;
    let num_shots = 8000;

    let unitary = damped_two_block_circuit(0);
    let mut measured = damped_two_block_circuit(4);
    for q in 0..4 {
        measured.add_measure(q, q);
    }

    let reference_noise = NoiseModel::with_amplitude_damping(&unitary, gamma);
    let observables: Vec<Vec<PauliTerm>> = (0..4).map(|q| vec![PauliTerm::z(q)]).collect();
    let exact =
        density_matrix_expectation_values(&unitary, &observables, Some(&reference_noise), 42)
            .unwrap();

    let noise = NoiseModel::with_amplitude_damping(&measured, gamma);
    let result =
        run_shots_with_noise(BackendKind::Statevector, &measured, &noise, num_shots, 42).unwrap();

    for q in 0..4 {
        let ones = result.shots.iter().filter(|shot| shot[q]).count();
        let measured_z = 1.0 - 2.0 * (ones as f64 / num_shots as f64);
        let p_exact = (1.0 - exact[q]) / 2.0;
        let sigma = (2.0 * (p_exact * (1.0 - p_exact) / num_shots as f64).sqrt()).max(1e-9);

        assert!(
            (measured_z - exact[q]).abs() <= 5.0 * sigma,
            "statevector trajectory <Z_{q}> = {measured_z}, density matrix = {}",
            exact[q]
        );
    }
}

// Thermal relaxation on the factored backend routes through `reset` and the
// dephasing branch rather than a Kraus matrix, so it exercises the other half
// of the non-Pauli path on the same two-block state.
#[test]
fn factored_thermal_relaxation_matches_density_matrix() {
    let num_shots = 8000;
    let channel = NoiseChannel::ThermalRelaxation {
        t1: 100.0,
        t2: 80.0,
        gate_time: 20.0,
    };

    let unitary = damped_two_block_circuit(0);
    let mut measured = damped_two_block_circuit(4);
    for q in 0..4 {
        measured.add_measure(q, q);
    }

    let thermal_model = |circuit: &Circuit| {
        let mut model = NoiseModel::uniform_depolarizing(circuit, 0.0);
        for (index, instruction) in circuit.instructions.iter().enumerate() {
            if let Instruction::Gate { targets, .. } = instruction {
                model.after_gate[index] = targets
                    .iter()
                    .map(|&q| NoiseEvent {
                        channel: channel.clone(),
                        qubits: SmallVec::from_slice(&[q]),
                    })
                    .collect();
            }
        }
        model
    };

    let observables: Vec<Vec<PauliTerm>> = (0..4).map(|q| vec![PauliTerm::z(q)]).collect();
    let exact = density_matrix_expectation_values(
        &unitary,
        &observables,
        Some(&thermal_model(&unitary)),
        42,
    )
    .unwrap();

    let result = run_shots_with_noise(
        BackendKind::Factored,
        &measured,
        &thermal_model(&measured),
        num_shots,
        42,
    )
    .unwrap();

    for q in 0..4 {
        let ones = result.shots.iter().filter(|shot| shot[q]).count();
        let measured_z = 1.0 - 2.0 * (ones as f64 / num_shots as f64);
        let p_exact = (1.0 - exact[q]) / 2.0;
        let sigma = (2.0 * (p_exact * (1.0 - p_exact) / num_shots as f64).sqrt()).max(1e-9);

        assert!(
            (measured_z - exact[q]).abs() <= 5.0 * sigma,
            "factored thermal <Z_{q}> = {measured_z}, density matrix = {}",
            exact[q]
        );
    }
}

#[test]
fn noise_model_validate_catches_invalid_custom_channel() {
    use num_complex::Complex64;

    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::Id, &[0]);
    circuit.add_measure(0, 0);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0] = vec![NoiseEvent {
        channel: NoiseChannel::Custom { kraus: Vec::new() },
        qubits: SmallVec::from_slice(&[0]),
    }];

    assert!(noise.validate().is_err());
    assert!(!noise.after_gate[0][0].channel.is_exactly_samplable());

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    noise.after_gate[0][0].channel = NoiseChannel::Custom {
        kraus: vec![[[one, zero], [zero, one]]],
    };

    assert!(noise.validate().is_ok());
    assert!(noise.after_gate[0][0].channel.is_exactly_samplable());
}
