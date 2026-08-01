use super::*;
use crate::circuits;

type EngineFn = fn(&Circuit, &NoiseModel, usize, u64) -> Result<ShotsResult>;

fn assert_ghz_noise_spread(run: EngineFn, n: usize, min_each: usize) {
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let result = run(&circuit, &noise, 1000, 42).unwrap();

    assert_eq!(result.shots.len(), 1000);
    assert_eq!(result.shots[0].len(), n);

    let all_zero: Vec<bool> = vec![false; n];
    let all_one: Vec<bool> = vec![true; n];
    let num_00 = result.shots.iter().filter(|s| **s == all_zero).count();
    let num_11 = result.shots.iter().filter(|s| **s == all_one).count();

    assert!(
        1000 - num_00 - num_11 > 0,
        "noise should produce non-GHZ outcomes"
    );
    assert!(
        num_00 > min_each,
        "should still have many |00...0> outcomes"
    );
    assert!(
        num_11 > min_each,
        "should still have many |11...1> outcomes"
    );
}

fn assert_ghz_zero_noise_coherent(run: EngineFn, n: usize) {
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    let result = run(&circuit, &noise, 100, 42).unwrap();
    assert_eq!(
        result.coherent_fraction(),
        1.0,
        "GHZ with zero noise must be all-0 or all-1"
    );
}

fn assert_clifford_noise_varies(run: EngineFn, n: usize) {
    let mut circuit = circuits::clifford_heavy_circuit(n, 10, 42);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let result = run(&circuit, &noise, 100, 42).unwrap();

    assert_eq!(result.shots.len(), 100);
    assert_eq!(result.shots[0].len(), n);

    let unique: std::collections::HashSet<Vec<bool>> = result.shots.iter().cloned().collect();
    assert!(unique.len() > 1, "noise should produce varied outcomes");
}

#[test]
fn noisy_ghz_produces_varied_outcomes() {
    assert_ghz_noise_spread(run_shots_noisy, 10, 100);
}

#[test]
fn zero_noise_matches_noiseless() {
    assert_ghz_zero_noise_coherent(run_shots_noisy, 5);
}

#[test]
fn noise_model_length_matches_circuit() {
    let n = 10;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    assert_eq!(noise.after_gate.len(), circuit.instructions.len());
}

#[test]
fn compiled_noisy_stats_match_brute_force() {
    let n = 10;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let num_shots = 10000;

    let brute = run_shots_noisy_brute_with(
        |s| Box::new(StabilizerBackend::new(s)),
        &circuit,
        &noise,
        num_shots,
        42,
    )
    .unwrap();
    let compiled = run_shots_noisy_compiled(&circuit, &noise, num_shots, 42).unwrap();

    let brute_frac = brute.coherent_fraction();
    let compiled_frac = compiled.coherent_fraction();

    assert!(
        (brute_frac - compiled_frac).abs() < 0.05,
        "coherent fraction should be similar: brute={brute_frac:.3}, compiled={compiled_frac:.3}"
    );
    assert!(
        brute_frac < 1.0 && compiled_frac < 1.0,
        "both should produce errors"
    );
}

#[test]
fn pauli_engines_share_observable_statistics() {
    let n = 6;
    let mut circuit = Circuit::new(n, n);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_gate(Gate::Cx, &[3, 4]);
    circuit.add_gate(Gate::H, &[5]);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
    let num_shots = 10_000;
    let seed = 42;

    let analytic =
        crate::sim::homological::noisy_marginals_analytical(&circuit, &noise, seed).unwrap();

    let sv_factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::statevector::StatevectorBackend::new(s))
    };
    let engines: Vec<(&str, ShotsResult)> = vec![
        (
            "compiled",
            run_shots_noisy_compiled(&circuit, &noise, num_shots, seed).unwrap(),
        ),
        (
            "frame",
            run_shots_noisy_frame(&circuit, &noise, num_shots, seed).unwrap(),
        ),
        (
            "homological",
            crate::sim::homological::run_shots_homological(&circuit, &noise, num_shots, seed)
                .unwrap(),
        ),
        (
            "brute",
            run_shots_noisy_brute_with(
                |s| Box::new(StabilizerBackend::new(s)),
                &circuit,
                &noise,
                num_shots,
                seed,
            )
            .unwrap(),
        ),
        (
            "trajectory",
            crate::sim::trajectory::run_trajectories(
                sv_factory, &circuit, &noise, num_shots, seed, false,
            )
            .unwrap(),
        ),
    ];

    // 5 sigma at 10k shots: sigma <= 0.5 / sqrt(N) = 0.005.
    let marginal_tol = 0.025;
    for (name, result) in &engines {
        assert_eq!(result.shots.len(), num_shots);
        for (bit, &expected) in analytic.iter().enumerate() {
            let p = result.marginal(bit);
            assert!(
                (p - expected).abs() < marginal_tol,
                "{name} bit {bit}: marginal {p:.4} vs analytic {expected:.4}"
            );
        }
    }

    // The density-matrix backend carries no sampling noise, so it sits far
    // tighter to the analytic marginals than the sampled engines. The
    // residual is a model gap, not statistics: the analytic reference is the
    // Pauli-frame independent-error model (X and Y flips are independent
    // Bernoulli events), while the density matrix evolves the exact
    // depolarizing channel (mutually exclusive Paulis). The two agree to
    // first order in p and differ at O(p^2); at p = 0.02 that gap is < 3e-4.
    let dm_marginals = dm_noisy_marginals(&circuit, &noise, seed).unwrap();
    let dm_tol = 3e-4;
    for (bit, &expected) in analytic.iter().enumerate() {
        assert!(
            (dm_marginals[bit] - expected).abs() < dm_tol,
            "density_matrix bit {bit}: marginal {:.12} vs analytic {expected:.12}",
            dm_marginals[bit]
        );
    }

    let ghz_coherent = |result: &ShotsResult| -> f64 {
        result
            .shots
            .iter()
            .filter(|s| s[2] == s[3] && s[3] == s[4])
            .count() as f64
            / num_shots as f64
    };
    let correlator_tol = 0.04;
    let reference = ghz_coherent(&engines[0].1);
    for (name, result) in &engines[1..] {
        let value = ghz_coherent(result);
        assert!(
            (value - reference).abs() < correlator_tol,
            "{name} GHZ coherence {value:.4} vs compiled {reference:.4}"
        );
    }
}

#[test]
fn dm_expectation_matches_run_expectation_values() {
    use crate::{PauliAxis, PauliTerm};
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Ry(0.7), &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Rx(1.1), &[2]);
    circuit.add_gate(Gate::Cz, &[1, 2]);
    circuit.add_gate(Gate::T, &[0]);

    let observables = vec![
        vec![PauliTerm::new(0, PauliAxis::Z)],
        vec![PauliTerm::new(2, PauliAxis::X)],
        vec![
            PauliTerm::new(0, PauliAxis::Z),
            PauliTerm::new(1, PauliAxis::Z),
        ],
        vec![
            PauliTerm::new(0, PauliAxis::Y),
            PauliTerm::new(2, PauliAxis::X),
        ],
    ];

    let reference = crate::sim::run_expectation_values(&circuit, &observables, 42).unwrap();
    let dm = density_matrix_expectation_values(&circuit, &observables, None, 42).unwrap();
    for (i, (a, b)) in dm.iter().zip(&reference).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "observable {i}: dm {a:.15} vs statevector {b:.15}"
        );
    }
}

#[test]
fn trajectory_expectations_converge_to_density_matrix_within_5_sigma() {
    use crate::backend::statevector::StatevectorBackend;
    use crate::sim::trajectory::run_trajectory_shot;
    use crate::{PauliAxis, PauliTerm};

    // Six-qubit seeded layered circuit with two noise layers.
    let mut circuit = Circuit::new(6, 0);
    for q in 0..6 {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..5 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
    for q in 0..6 {
        let theta: f64 = rand::RngExt::random::<f64>(&mut rng) * std::f64::consts::TAU;
        circuit.add_gate(
            if q % 2 == 0 {
                Gate::Rz(theta)
            } else {
                Gate::Rx(theta)
            },
            &[q],
        );
    }

    // Noise after layer 1 (the first H) and layer 2 (first CX).
    let mut noise = NoiseModel {
        after_gate: vec![Vec::new(); circuit.instructions.len()],
        readout: vec![None; circuit.num_classical_bits],
    };
    for q in 0..6 {
        noise.after_gate[q].push(NoiseEvent {
            channel: NoiseChannel::Depolarizing { p: 0.02 },
            qubits: smallvec![q],
        });
    }
    for q in 0..6 {
        noise.after_gate[6 + q.min(4)].push(NoiseEvent {
            channel: NoiseChannel::AmplitudeDamping { gamma: 0.03 },
            qubits: smallvec![q],
        });
    }

    let observables = vec![
        vec![
            PauliTerm::new(0, PauliAxis::Z),
            PauliTerm::new(1, PauliAxis::Z),
        ],
        vec![
            PauliTerm::new(2, PauliAxis::X),
            PauliTerm::new(3, PauliAxis::X),
        ],
        vec![
            PauliTerm::new(4, PauliAxis::Z),
            PauliTerm::new(5, PauliAxis::Z),
        ],
    ];
    let weights = [0.5f64, 0.3, 0.2];

    let mu_dm =
        density_matrix_expectation_values(&circuit, &observables, Some(&noise), 42).unwrap();

    // Trajectory estimates: each shot is a pure state; the observable value
    // on that state is a sample whose mean converges to Tr(rho P).
    let masks: Vec<_> = observables
        .iter()
        .map(|obs| crate::sim::pauli_masks(obs, circuit.num_qubits).unwrap())
        .collect();
    let shots = 20_000usize;
    let mut sums = vec![0.0f64; observables.len()];
    let mut sq_sums = vec![0.0f64; observables.len()];
    let mut backend = StatevectorBackend::new(42);
    for i in 0..shots {
        let mut shot_rng = ChaCha8Rng::seed_from_u64(42u64.wrapping_add(i as u64));
        run_trajectory_shot(&mut backend, &circuit, &noise, &mut shot_rng).unwrap();
        let state = backend.state_vector();
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        for (k, &(x, z, y)) in masks.iter().enumerate() {
            let v = crate::sim::pauli_expectation_from_masks(state, x, z, y, norm);
            sums[k] += v;
            sq_sums[k] += v * v;
        }
    }

    for (k, &w) in weights.iter().enumerate() {
        let mean = sums[k] / shots as f64;
        let var = (sq_sums[k] / shots as f64 - mean * mean).max(0.0);
        let sigma = (var / shots as f64).sqrt();
        let diff = (mean - mu_dm[k]).abs();
        if sigma < 1e-12 {
            assert!(
                diff < 1e-10,
                "term {k} (w={w}): degenerate trajectory, mean {mean} vs dm {}",
                mu_dm[k]
            );
        } else {
            let z_score = diff / sigma;
            assert!(
                z_score < 5.0,
                "term {k} (w={w}): z={z_score:.2}, traj {mean:.6} vs dm {:.6}",
                mu_dm[k]
            );
        }
    }
}

#[test]
fn dm_named_channel_lowering_preserves_trace_and_decay() {
    use crate::backend::density_matrix::DensityMatrixBackend;

    let trace = |dm: &DensityMatrixBackend| -> f64 { dm.probabilities().unwrap().iter().sum() };
    let identity = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let channels = [
        NoiseChannel::Pauli {
            px: 0.1,
            py: 0.05,
            pz: 0.2,
        },
        NoiseChannel::Depolarizing { p: 0.3 },
        NoiseChannel::AmplitudeDamping { gamma: 0.4 },
        NoiseChannel::PhaseDamping { gamma: 0.35 },
        NoiseChannel::ThermalRelaxation {
            t1: 50.0,
            t2: 40.0,
            gate_time: 10.0,
        },
        NoiseChannel::Custom {
            kraus: vec![identity],
        },
    ];
    for ch in &channels {
        let mut dm = DensityMatrixBackend::new(42);
        dm.init(1, 0).unwrap();
        dm.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap();
        dm.apply_1q_kraus(0, &kraus_1q(ch));
        assert!(
            (trace(&dm) - 1.0).abs() < 1e-12,
            "trace after {ch:?}: {}",
            trace(&dm)
        );
    }

    let (t1, t2, gt) = (50.0, 40.0, 10.0);
    let thermal = NoiseChannel::ThermalRelaxation {
        t1,
        t2,
        gate_time: gt,
    };
    let mut dm = DensityMatrixBackend::new(42);
    dm.init(1, 0).unwrap();
    dm.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: smallvec![0],
    })
    .unwrap();
    dm.apply_1q_kraus(0, &kraus_1q(&thermal));
    let rho = dm.reduced_density_matrix_1q(0).unwrap();
    assert!(
        (rho[1][1].re - (-gt / t1).exp()).abs() < 1e-12,
        "T1 population decay: {rho:?}"
    );

    let mut dm = DensityMatrixBackend::new(42);
    dm.init(1, 0).unwrap();
    dm.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec![0],
    })
    .unwrap();
    dm.apply_1q_kraus(0, &kraus_1q(&thermal));
    let rho = dm.reduced_density_matrix_1q(0).unwrap();
    assert!(
        (rho[0][1].norm() - 0.5 * (-gt / t2).exp()).abs() < 1e-12,
        "T2 coherence decay: {rho:?}"
    );

    // Amplitude and phase damping lowerings drive the exact analytic channel
    // through kraus_1q, not just trace preservation, on |+>.
    let prep_plus = |ch: &NoiseChannel| -> [[Complex64; 2]; 2] {
        let mut dm = DensityMatrixBackend::new(42);
        dm.init(1, 0).unwrap();
        dm.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap();
        dm.apply_1q_kraus(0, &kraus_1q(ch));
        dm.reduced_density_matrix_1q(0).unwrap()
    };

    let gamma = 0.4;
    let rho = prep_plus(&NoiseChannel::AmplitudeDamping { gamma });
    assert!(
        (rho[0][0].re - (0.5 + 0.5 * gamma)).abs() < 1e-12,
        "AD pop0: {rho:?}"
    );
    assert!(
        (rho[1][1].re - 0.5 * (1.0 - gamma)).abs() < 1e-12,
        "AD pop1: {rho:?}"
    );
    assert!(
        (rho[0][1].norm() - 0.5 * (1.0 - gamma).sqrt()).abs() < 1e-12,
        "AD coherence: {rho:?}"
    );

    let gamma = 0.35;
    let rho = prep_plus(&NoiseChannel::PhaseDamping { gamma });
    assert!(
        (rho[0][0].re - 0.5).abs() < 1e-12,
        "PD pop0 preserved: {rho:?}"
    );
    assert!(
        (rho[1][1].re - 0.5).abs() < 1e-12,
        "PD pop1 preserved: {rho:?}"
    );
    assert!(
        (rho[0][1].norm() - 0.5 * (1.0 - gamma).sqrt()).abs() < 1e-12,
        "PD coherence: {rho:?}"
    );
}

#[test]
fn compiled_noisy_clifford_produces_noise() {
    assert_clifford_noise_varies(run_shots_noisy, 20);
}

#[test]
fn compile_noisy_rejects_reset_circuits() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    assert!(compile_noisy(&circuit, &noise, 42).is_err());
}

#[test]
fn compile_noisy_rejects_conditionals() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: crate::circuit::smallvec![1],
    });
    circuit.add_measure(1, 1);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    assert!(compile_noisy(&circuit, &noise, 42).is_err());
}

#[test]
fn run_shots_noisy_handles_reset_circuits() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    let result = run_shots_noisy(&circuit, &noise, 32, 42).unwrap();
    assert!(result.shots.iter().all(|shot| !shot[0]));
}

#[test]
fn run_shots_noisy_handles_conditionals() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: crate::circuit::smallvec![1],
    });
    circuit.add_measure(1, 1);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    let result = run_shots_noisy(&circuit, &noise, 256, 42).unwrap();
    assert!(result.shots.iter().all(|shot| shot[0] == shot[1]));
}

#[test]
fn frame_ghz_100q_produces_varied_outcomes() {
    assert_ghz_noise_spread(run_shots_noisy_frame, 100, 50);
}

#[test]
fn frame_zero_noise_matches_noiseless_100q() {
    assert_ghz_zero_noise_coherent(run_shots_noisy_frame, 100);
}

#[test]
fn frame_stats_match_compiled_ghz() {
    let n = 100;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let num_shots = 5000;

    let frame = run_shots_noisy_frame(&circuit, &noise, num_shots, 42).unwrap();
    let compiled = run_shots_noisy_compiled(&circuit, &noise, num_shots, 42).unwrap();

    let frame_coh = frame.coherent_fraction();
    let compiled_coh = compiled.coherent_fraction();

    assert!(
        (frame_coh - compiled_coh).abs() < 0.05,
        "coherent fraction should be similar: frame={frame_coh:.3}, compiled={compiled_coh:.3}"
    );
}

#[test]
fn frame_clifford_100q_produces_noise() {
    assert_clifford_noise_varies(run_shots_noisy_frame, 100);
}

#[test]
fn filtered_noisy_bell_pairs_matches_monolithic() {
    let n_pairs = 50;
    let mut circuit = circuits::independent_bell_pairs(n_pairs);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let seed = 42u64;

    let filtered =
        compile_noisy_filtered(&circuit, &noise, &circuit.independent_subsystems(), seed).unwrap();
    let monolithic = compile_noisy_monolithic(&circuit, &noise, seed).unwrap();

    assert_eq!(filtered.num_measurements, monolithic.num_measurements);
    assert_eq!(filtered.events.len(), monolithic.events.len());

    let mut filtered = filtered;
    let mut monolithic = monolithic;
    let num_shots = 10_000;
    let shots_f = filtered.sample_bulk(num_shots);
    let shots_m = monolithic.sample_bulk(num_shots);

    assert_eq!(shots_f.len(), num_shots);
    assert_eq!(shots_m.len(), num_shots);

    let mut agree_f = 0usize;
    let mut agree_m = 0usize;
    for shot in &shots_f {
        for pair in shot.chunks(2) {
            if pair[0] == pair[1] {
                agree_f += 1;
            }
        }
    }
    for shot in &shots_m {
        for pair in shot.chunks(2) {
            if pair[0] == pair[1] {
                agree_m += 1;
            }
        }
    }

    let total_pairs = num_shots * n_pairs;
    let agree_rate_f = agree_f as f64 / total_pairs as f64;
    let agree_rate_m = agree_m as f64 / total_pairs as f64;
    assert!(
        agree_rate_f > 0.95,
        "filtered agreement rate {agree_rate_f:.4} should be >0.95 with low noise"
    );
    assert!(
        agree_rate_m > 0.95,
        "monolithic agreement rate {agree_rate_m:.4} should be >0.95 with low noise"
    );
    assert!(
        (agree_rate_f - agree_rate_m).abs() < 0.02,
        "filtered ({agree_rate_f:.4}) and monolithic ({agree_rate_m:.4}) should have similar agreement rates"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn noisy_gpu_test_circuit_routes_to_gpu_bts() {
    let mut circuit = Circuit::new(32, 32);
    for q in 0..16 {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 16..32 {
        for k in 0..4 {
            circuit.add_gate(Gate::Cx, &[(q - 16 + k) % 16, q]);
        }
    }
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);

    let gpu = compile_noisy(&circuit, &noise, 42)
        .unwrap()
        .with_gpu(crate::gpu::GpuContext::stub_for_tests());
    assert!(
        gpu.noiseless
            .should_use_gpu_bts(crate::gpu::bts_min_shots()),
        "the golden_gpu noisy reduction test relies on this circuit shape \
         passing the GPU BTS routing gates"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn compiled_noisy_with_stub_gpu_matches_cpu_below_threshold() {
    let n = 12;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let shots = 20_000;

    let mut cpu = compile_noisy(&circuit, &noise, 42).unwrap();
    let cpu_marginals = cpu.sample_marginals(shots);

    let mut gpu = compile_noisy(&circuit, &noise, 42)
        .unwrap()
        .with_gpu(crate::gpu::GpuContext::stub_for_tests());
    let gpu_marginals = gpu.sample_marginals(shots);

    for (idx, (cpu_p1, gpu_p1)) in cpu_marginals.iter().zip(gpu_marginals.iter()).enumerate() {
        assert!(
            (cpu_p1 - gpu_p1).abs() < 0.03,
            "marginal[{idx}] diverged too much: cpu={cpu_p1}, gpu={gpu_p1}"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn compiled_noisy_with_stub_gpu_low_rank_above_threshold_uses_cpu_fallback() {
    let shots = crate::gpu::bts_min_shots().max(1);
    let n = 12;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let mut gpu = compile_noisy(&circuit, &noise, 42)
        .unwrap()
        .with_gpu(crate::gpu::GpuContext::stub_for_tests());

    assert!(!gpu.noiseless.should_use_gpu_bts(shots));
    let counts = gpu.sample_counts(shots);
    let total: u64 = counts.values().sum();
    assert_eq!(total, shots as u64);
}
