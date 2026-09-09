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
                sv_factory,
                &circuit,
                &noise,
                num_shots,
                seed,
                false,
                crate::sim::ResolvedBackend::Statevector,
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
    let readout = crate::sim::trajectory::written_readout(&circuit, &noise.readout);
    for i in 0..shots {
        let mut shot_rng = ChaCha8Rng::seed_from_u64(42u64.wrapping_add(i as u64));
        run_trajectory_shot(&mut backend, &circuit, &noise, &readout, &mut shot_rng).unwrap();
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

// The frame and compiled routes answer from the same distribution and stamp the
// same resolved backend, so nothing in a result separates them. The route is
// pinned here through `use_frame_sampler`, in the same test that reads the
// rates.
fn readout_route_circuit(pad: usize) -> Circuit {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[1]);
    for _ in 0..pad {
        circuit.add_gate(Gate::Z, &[2]);
    }
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);
    circuit
}

fn bit_rate(shots: &[Vec<bool>], bit: usize, want: bool) -> f64 {
    shots.iter().filter(|s| s[bit] == want).count() as f64 / shots.len() as f64
}

// Rates far apart in both directions: applied before the reference bits are
// folded in, bit 0 would read p01 = 0.40 where p10 = 0.05 belongs. The bound is
// five sigma on a rate estimated from 20k draws, rounded up.
fn assert_asymmetric_readout_rates(circuit: &Circuit) {
    let mut noise = NoiseModel::uniform_depolarizing(circuit, 0.0);
    noise.set_bit_readout_error(0, 0.40, 0.05);
    noise.set_bit_readout_error(2, 0.20, 0.60);

    let result = crate::sim::simulate(circuit)
        .noise(&noise)
        .seed(42)
        .shots(20_000)
        .unwrap();

    let flipped_one = bit_rate(&result.shots, 0, false);
    assert!(
        (flipped_one - 0.05).abs() < 0.02,
        "bit 0 measures 1 and takes p10 = 0.05, got {flipped_one}"
    );
    let flipped_zero = bit_rate(&result.shots, 2, true);
    assert!(
        (flipped_zero - 0.20).abs() < 0.02,
        "bit 2 measures 0 and takes p01 = 0.20, got {flipped_zero}"
    );
    assert!(
        result.shots.iter().all(|s| s[1]),
        "bit 1 carries no readout entry and must keep its noiseless value"
    );
}

#[test]
fn readout_rates_hold_on_the_frame_route() {
    let circuit = readout_route_circuit(0);
    assert!(use_frame_sampler(&circuit));
    assert_asymmetric_readout_rates(&circuit);
}

#[test]
fn readout_rates_hold_on_the_compiled_route() {
    let circuit = readout_route_circuit(10);
    assert!(!use_frame_sampler(&circuit));
    assert_asymmetric_readout_rates(&circuit);
}

fn entangled_route_circuit(pad: usize) -> Circuit {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    for _ in 0..pad {
        circuit.add_gate(Gate::Z, &[2]);
    }
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);
    circuit
}

// The trajectory engine draws readout per shot against the unpacked record,
// which shares no code with the packed walk. Per-bit sigma is at most 0.0035 at
// 20k draws and the difference of two such estimates at most 0.005, so 0.03 is
// six sigma on the difference.
fn assert_matches_trajectory_engine(circuit: &Circuit) {
    let mut noise = NoiseModel::uniform_depolarizing(circuit, 0.02);
    noise.set_bit_readout_error(0, 0.10, 0.30);
    noise.set_bit_readout_error(1, 0.25, 0.05);
    noise.set_bit_readout_error(2, 0.40, 0.10);

    let clifford = crate::sim::simulate(circuit)
        .noise(&noise)
        .seed(42)
        .shots(20_000)
        .unwrap();
    assert_eq!(
        clifford.metadata.backend,
        crate::sim::ResolvedBackend::CompiledStabilizer
    );
    let trajectory = crate::sim::simulate(circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(42)
        .shots(20_000)
        .unwrap();
    assert_eq!(
        trajectory.metadata.backend,
        crate::sim::ResolvedBackend::Statevector
    );

    for bit in 0..3 {
        let c = bit_rate(&clifford.shots, bit, true);
        let t = bit_rate(&trajectory.shots, bit, true);
        assert!(
            (c - t).abs() < 0.03,
            "bit {bit}: clifford {c:.4} vs trajectory {t:.4}"
        );
    }
}

#[test]
fn readout_matches_the_trajectory_engine_on_the_frame_route() {
    let circuit = entangled_route_circuit(0);
    assert!(use_frame_sampler(&circuit));
    assert_matches_trajectory_engine(&circuit);
}

#[test]
fn readout_matches_the_trajectory_engine_on_the_compiled_route() {
    let circuit = entangled_route_circuit(7);
    assert!(!use_frame_sampler(&circuit));
    assert_matches_trajectory_engine(&circuit);
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

// The identity product is not a branch: an index of 0 would leave 1 firing
// shot in 15 unchanged and cut the effective rate by exactly that much.
#[test]
fn pair_branch_covers_the_fifteen_non_identity_products() {
    let mut seen = [0usize; 16];
    for i in 0..15_000 {
        let (l0, l1) = pair_branch(i as f64 / 15_000.0);
        seen[l0 * 4 + l1] += 1;
    }
    assert_eq!(seen[0], 0, "identity is not a branch");
    assert!(seen[1..].iter().all(|&count| count == 1000), "{seen:?}");
}

fn pair_event(qubits: [usize; 2], p: f64) -> NoiseEvent {
    NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p },
        qubits: qubits.into_iter().collect(),
    }
}

fn bare_model(circuit: &Circuit) -> NoiseModel {
    NoiseModel {
        after_gate: vec![Vec::new(); circuit.instructions.len()],
        readout: vec![None; circuit.num_classical_bits],
    }
}

/// `X q0` then a CX chain, so the ideal record is all ones and every
/// single-qubit Z moves with the noise. `pad` is a run of Z on qubit 0, which
/// no Z-basis outcome sees and which moves the gate count per qubit across the
/// frame cutoff.
fn pair_chain_circuit(n: usize, pad: usize) -> Circuit {
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

fn pair_after_every_cx(circuit: &Circuit, p: f64) -> NoiseModel {
    let mut model = bare_model(circuit);
    for (slot, inst) in model.after_gate.iter_mut().zip(&circuit.instructions) {
        if let Instruction::Gate {
            gate: Gate::Cx,
            targets,
        } = inst
        {
            slot.push(pair_event([targets[0], targets[1]], p));
        }
    }
    model
}

fn chain_observables(n: usize) -> Vec<Vec<crate::PauliTerm>> {
    let mut observables: Vec<Vec<crate::PauliTerm>> =
        (0..n).map(|q| vec![crate::PauliTerm::z(q)]).collect();
    observables
        .extend((0..n - 1).map(|q| vec![crate::PauliTerm::z(q), crate::PauliTerm::z(q + 1)]));
    observables
}

fn sampled_chain_expectations(shots: &[Vec<bool>], n: usize) -> Vec<f64> {
    let total = shots.len() as f64;
    let mut got: Vec<f64> = (0..n)
        .map(|q| 1.0 - 2.0 * shots.iter().filter(|s| s[q]).count() as f64 / total)
        .collect();
    got.extend(
        (0..n - 1)
            .map(|q| 1.0 - 2.0 * shots.iter().filter(|s| s[q] != s[q + 1]).count() as f64 / total),
    );
    got
}

/// Five sigma on a Z expectation estimated from 20k draws at the worst-case
/// variance, rounded up.
const CHAIN_BAND: f64 = 0.036;

fn assert_chain_matches_density_matrix(circuit: &Circuit, noise: &NoiseModel, n: usize) {
    let result = crate::sim::simulate(circuit)
        .noise(noise)
        .seed(42)
        .shots(20_000)
        .unwrap();
    let want =
        density_matrix_expectation_values(circuit, &chain_observables(n), Some(noise), 42).unwrap();

    for (i, (g, w)) in sampled_chain_expectations(&result.shots, n)
        .iter()
        .zip(&want)
        .enumerate()
    {
        assert!(
            (g - w).abs() < CHAIN_BAND,
            "observable {i}: sampled {g:.4} against density matrix {w:.4}"
        );
    }
}

// Every qubit accumulates one letter per event that reaches it, so a single
// qubit Z here separates the joint draw from an implementation that resolves
// only the first target. The nearest-neighbour parities cannot: each is fed by
// one letter of each of two events, and dropping the second leaves the same
// marginal.
#[test]
fn pair_channel_matches_the_density_matrix_on_the_frame_route() {
    let circuit = pair_chain_circuit(8, 0);
    assert!(use_frame_sampler(&circuit));
    assert_chain_matches_density_matrix(&circuit, &pair_after_every_cx(&circuit, 0.05), 8);
}

#[test]
fn pair_channel_matches_the_density_matrix_on_the_compiled_route() {
    let circuit = pair_chain_circuit(8, 18);
    assert!(!use_frame_sampler(&circuit));
    assert_chain_matches_density_matrix(&circuit, &pair_after_every_cx(&circuit, 0.05), 8);
}

// The grouped route XORs the single-qubit rows through a flip lookup table
// while the pair table runs as a second pass over the same buffer, so a model
// carrying both is the only thing that exercises the two together.
#[test]
fn a_mixed_model_runs_the_flip_table_and_the_pair_pass() {
    let n = 9;
    let circuit = pair_chain_circuit(n, 18);
    assert!(!use_frame_sampler(&circuit));

    let mut noise = pair_after_every_cx(&circuit, 0.02);
    for (slot, inst) in noise.after_gate.iter_mut().zip(&circuit.instructions) {
        if let Instruction::Gate {
            gate: Gate::X | Gate::Cx,
            targets,
        } = inst
        {
            slot.extend(targets.iter().map(|&q| NoiseEvent {
                channel: NoiseChannel::Depolarizing { p: 0.055 },
                qubits: smallvec![q],
            }));
        }
    }

    let sampler = compile_noisy(&circuit, &noise, 42).unwrap();
    assert!(sampler.z_lut.is_some(), "the flip table must be built");
    assert!(sampler.events.has_pairs());

    assert_chain_matches_density_matrix(&circuit, &noise, n);
}

/// `|++>` under one two-qubit channel, read back in the X basis. `pad` is an
/// even run of H on qubit 0, which leaves the state alone and moves the gate
/// count per qubit across the frame cutoff.
fn pair_x_basis_circuit(pad: usize) -> Circuit {
    let mut circuit = Circuit::new(2, 2);
    for _ in 0..pad {
        circuit.add_gate(Gate::H, &[0]);
    }
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    circuit
}

fn x_basis_pair_model(circuit: &Circuit, pad: usize, p: f64) -> NoiseModel {
    let mut model = bare_model(circuit);
    model.after_gate[pad + 1] = vec![pair_event([0, 1], p)];
    model
}

// In the X basis a bit reads 1 under Y or Z, so the outcome separates the 15
// branches into 3 that leave both bits at 0 and 4 that set both. Running the
// branch index 0 through 15 instead of 1 through 15 swaps identity in for ZZ
// and swaps those two rates. No Z-basis statistic can see that swap, because
// identity and ZZ flip the same measurements, namely none.
fn assert_identity_is_not_a_branch(circuit: &Circuit, pad: usize) {
    let noise = x_basis_pair_model(circuit, pad, 1.0);
    let result = crate::sim::simulate(circuit)
        .noise(&noise)
        .seed(42)
        .shots(20_000)
        .unwrap();

    let total = result.shots.len() as f64;
    let both_zero = result.shots.iter().filter(|s| !s[0] && !s[1]).count() as f64 / total;
    let both_one = result.shots.iter().filter(|s| s[0] && s[1]).count() as f64 / total;
    assert!(
        (both_zero - 3.0 / 15.0).abs() < 0.02,
        "3 of the 15 branches leave both bits at 0, got {both_zero}"
    );
    assert!(
        (both_one - 4.0 / 15.0).abs() < 0.02,
        "4 of the 15 branches set both bits, got {both_one}"
    );
}

#[test]
fn identity_is_not_a_branch_on_the_frame_route() {
    let circuit = pair_x_basis_circuit(0);
    assert!(use_frame_sampler(&circuit));
    assert_identity_is_not_a_branch(&circuit, 0);
}

#[test]
fn identity_is_not_a_branch_on_the_compiled_route() {
    let circuit = pair_x_basis_circuit(4);
    assert!(!use_frame_sampler(&circuit));
    assert_identity_is_not_a_branch(&circuit, 4);
}

// One branch of 15 decides both letters, so each of the three outcomes that
// moves at least one bit carries `4p/15 = 0.133`. Two independent draws
// matching the per-qubit rate of `8p/15` would put `(8p/15)^2 = 0.071` on the
// corner where both bits move, which is the whole difference between a
// correlated channel and two uncorrelated ones.
fn assert_the_two_letters_are_drawn_jointly(circuit: &Circuit) {
    let p = 0.5;
    let mut noise = bare_model(circuit);
    noise.after_gate[1] = vec![pair_event([0, 1], p)];

    let result = crate::sim::simulate(circuit)
        .noise(&noise)
        .seed(42)
        .shots(20_000)
        .unwrap();
    let total = result.shots.len() as f64;
    let rate = |want: [bool; 2]| {
        result
            .shots
            .iter()
            .filter(|s| s[0] == want[0] && s[1] == want[1])
            .count() as f64
            / total
    };

    let corner = 4.0 * p / 15.0;
    for (label, want, expected) in [
        ("both bits move", [false, false], corner),
        ("only qubit 0 moves", [false, true], corner),
        ("only qubit 1 moves", [true, false], corner),
        ("neither moves", [true, true], 1.0 - 3.0 * corner),
    ] {
        let got = rate(want);
        assert!(
            (got - expected).abs() < 0.02,
            "{label}: got {got:.4}, expected {expected:.4}"
        );
    }
}

#[test]
fn the_two_letters_are_drawn_jointly_on_the_frame_route() {
    let circuit = pair_chain_circuit(2, 0);
    assert!(use_frame_sampler(&circuit));
    assert_the_two_letters_are_drawn_jointly(&circuit);
}

#[test]
fn the_two_letters_are_drawn_jointly_on_the_compiled_route() {
    let circuit = pair_chain_circuit(2, 6);
    assert!(!use_frame_sampler(&circuit));
    assert_the_two_letters_are_drawn_jointly(&circuit);
}

// A zero-rate pair flips nothing, so it must not cost the model the device
// path the way a live one does, and must not block the homological gate, which
// otherwise reads the channel variant alone. This is the reading the readout
// table already uses: inert, not absent.
#[test]
fn a_zero_rate_pair_channel_is_inert() {
    let circuit = pair_x_basis_circuit(4);

    let inert = x_basis_pair_model(&circuit, 4, 0.0);
    let sampler = compile_noisy(&circuit, &inert, 42).unwrap();
    assert!(!sampler.events.has_pairs());
    assert!(inert.is_pauli_only());
    assert!(inert.ensure_pauli_only().is_ok());
    assert!(crate::sim::homological::noisy_marginals_analytical(&circuit, &inert, 42).is_ok());

    let live = x_basis_pair_model(&circuit, 4, 0.2);
    let sampler = compile_noisy(&circuit, &live, 42).unwrap();
    assert!(sampler.events.has_pairs());
    assert!(!live.is_pauli_only());
}

// The inert entry must also draw nothing, which only a circuit with a random
// outcome and a live channel alongside it can show: the two runs share one
// stream and diverge on the first extra draw.
#[test]
fn a_zero_rate_pair_channel_consumes_no_randomness() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    for _ in 0..6 {
        circuit.add_gate(Gate::Z, &[0]);
    }
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    assert!(!use_frame_sampler(&circuit));

    let singles = NoiseModel::uniform_depolarizing(&circuit, 0.08);
    let mut with_inert_pair = NoiseModel::uniform_depolarizing(&circuit, 0.08);
    with_inert_pair.after_gate[1].push(pair_event([0, 1], 0.0));

    let baseline = run_shots_noisy(&circuit, &singles, 4000, 42).unwrap();
    let inert = run_shots_noisy(&circuit, &with_inert_pair, 4000, 42).unwrap();
    assert!(
        baseline.shots.iter().any(|s| s[0]) && baseline.shots.iter().any(|s| !s[0]),
        "the fixture must carry a random outcome for the comparison to mean anything"
    );
    assert_eq!(baseline.shots, inert.shots);
}

/// Two subsystems that no gate couples, over a deterministic all-ones record.
fn two_block_circuit() -> Circuit {
    let mut circuit = Circuit::new(4, 4);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::X, &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    for q in 0..4 {
        circuit.add_measure(q, q);
    }
    circuit
}

// A pair whose two qubits land in different subsystem blocks has no single
// block to be propagated in, so the model takes the monolithic compile. A pair
// inside one block keeps the filtered one.
#[test]
fn a_block_straddling_pair_leaves_the_filtered_compile() {
    let circuit = two_block_circuit();
    let blocks = circuit.independent_subsystems();
    assert_eq!(blocks.len(), 2);

    let model = |qubits: [usize; 2], p: f64| {
        let mut noise = bare_model(&circuit);
        noise.after_gate[1] = vec![pair_event(qubits, p)];
        noise
    };

    assert!(pairs_stay_in_one_block(
        &circuit,
        &model([0, 1], 0.2),
        &blocks
    ));
    assert!(!pairs_stay_in_one_block(
        &circuit,
        &model([1, 2], 0.2),
        &blocks
    ));
    assert!(
        pairs_stay_in_one_block(&circuit, &model([1, 2], 0.0), &blocks),
        "a zero-rate pair stores no row and constrains no route"
    );
}

// The filtered compile scatters each block's local masks back into the global
// measurement order, four rows for a pair where a single-qubit event has two.
// Nothing else reaches that scatter carrying a live pair.
#[test]
fn the_filtered_compile_carries_an_in_block_pair() {
    let circuit = two_block_circuit();
    let blocks = circuit.independent_subsystems();
    let mut noise = bare_model(&circuit);
    noise.after_gate[1] = vec![pair_event([0, 1], 0.5)];
    assert!(pairs_stay_in_one_block(&circuit, &noise, &blocks));

    let mut sampler = compile_noisy_filtered(&circuit, &noise, &blocks, 42).unwrap();
    assert!(sampler.events.has_pairs());
    let shots = sampler.sample_bulk_packed(20_000).to_shots();

    let observables: Vec<Vec<crate::PauliTerm>> = (0..4)
        .map(|q| vec![crate::PauliTerm::z(q)])
        .chain(std::iter::once(vec![
            crate::PauliTerm::z(0),
            crate::PauliTerm::z(1),
        ]))
        .collect();
    let want = density_matrix_expectation_values(&circuit, &observables, Some(&noise), 42).unwrap();

    let total = shots.len() as f64;
    let mut got: Vec<f64> = (0..4)
        .map(|q| 1.0 - 2.0 * shots.iter().filter(|s| s[q]).count() as f64 / total)
        .collect();
    got.push(1.0 - 2.0 * shots.iter().filter(|s| s[0] != s[1]).count() as f64 / total);

    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < CHAIN_BAND,
            "observable {i}: sampled {g:.4} against density matrix {w:.4}"
        );
    }
}
