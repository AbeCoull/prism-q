use super::*;
use crate::circuits;

#[test]
fn thermal_trajectories_keep_the_seeded_records() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Ry(0.73), &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Rx(0.41), &[0]);
    circuit.measure_all();
    let factory = |seed| -> Box<dyn Backend> { Box::new(StatevectorBackend::new(seed)) };
    let mut records = Vec::new();
    for excited in [0.0, 0.3] {
        let thermal = |qubit, gate_time| NoiseEvent {
            channel: NoiseChannel::ThermalRelaxation {
                t1: 40.0,
                t2: 55.0,
                gate_time,
                excited_population: excited,
            },
            qubits: smallvec![qubit],
        };
        let mut noise = NoiseModel {
            after_gate: vec![Vec::new(); circuit.instructions.len()],
            readout: vec![
                Some(ReadoutError {
                    p01: 0.04,
                    p10: 0.07,
                }),
                None,
            ],
        };
        noise.after_gate[0].push(thermal(0, 10.0));
        noise.after_gate[1] = vec![
            NoiseEvent {
                channel: NoiseChannel::Depolarizing { p: 0.17 },
                qubits: smallvec![1],
            },
            thermal(1, f64::MIN_POSITIVE),
        ];
        noise.after_gate[2] = vec![
            thermal(0, 0.0),
            NoiseEvent {
                channel: NoiseChannel::AmplitudeDamping { gamma: 0.11 },
                qubits: smallvec![1],
            },
            thermal(1, 4.0),
        ];
        let run = |force_serial| {
            run_trajectories(
                factory,
                &circuit,
                &noise,
                128,
                42,
                force_serial,
                crate::sim::ResolvedBackend::Statevector,
            )
            .unwrap()
        };
        let serial = run(true);
        assert_eq!(run(false).shots, serial.shots);
        let mut words = [0u64; 4];
        for (i, &bit) in serial.shots.iter().flatten().enumerate() {
            words[i / 64] |= u64::from(bit) << (i % 64);
        }
        records.push(words);
    }
    assert_eq!(
        records,
        [
            [
                9224519968870264836,
                13848632576476766240,
                3801337310814801361,
                13890320543975999040,
            ],
            [
                9224520175028698884,
                14064805371743890976,
                17637534460143341009,
                13891164969043494464,
            ],
        ]
    );
}

#[test]
fn thermal_identity_channels_keep_the_draw_count() {
    for (t1, t2, gate_time, draws) in [
        (0.0, 1.0, 1.0, 0),
        (1.0, 0.0, 1.0, 0),
        (1.0, 1.0, 0.0, 0),
        (1.0, 1.0, f64::MIN_POSITIVE, 1),
    ] {
        let mut backend = StatevectorBackend::new(42);
        backend.init(1, 0).unwrap();
        let mut rng = noise_rng(42);
        let noise = NoiseModel {
            after_gate: vec![vec![NoiseEvent {
                channel: NoiseChannel::ThermalRelaxation {
                    t1,
                    t2,
                    gate_time,
                    excited_population: 0.0,
                },
                qubits: smallvec![0],
            }]],
            readout: Vec::new(),
        };
        let prepared = PreparedNoise::new(&noise);
        apply_noise_event(&mut backend, &prepared.slot(0)[0], &mut rng).unwrap();
        let mut expected = noise_rng(42);
        for _ in 0..draws {
            let _: f64 = rand::RngExt::random(&mut expected);
        }
        assert_eq!(
            rand::RngExt::random::<u64>(&mut rng),
            rand::RngExt::random::<u64>(&mut expected)
        );
    }
}

// The replica cap is only a memory bound if it stays at or below the cap
// governing a single state. Splitting statevector trajectories past the
// statevector cap would let the parallel path allocate one oversize state
// per thread, which is the failure `run_trajectories` avoids by falling
// back to serial execution.
#[cfg(feature = "parallel")]
#[test]
fn parallel_trajectory_replicas_stay_within_a_single_state_cap() {
    let cap = ResolvedLimits::process_default().statevector_qubits.value;
    let statevector = crate::sim::ResolvedBackend::Statevector;
    let replica_cap = (1..)
        .find(|&n| !crate::sim::state_splits_across_workers(statevector, n))
        .unwrap();
    assert!(
        replica_cap <= cap,
        "parallel replica cap {replica_cap} exceeds the statevector cap {cap}"
    );
}

#[test]
fn trajectory_pauli_matches_brute_force() {
    let n = 5;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
    let num_shots = 5000;

    let factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::statevector::StatevectorBackend::new(s))
    };
    let trajectory = run_trajectories(
        factory,
        &circuit,
        &noise,
        num_shots,
        42,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap();
    let brute = crate::sim::noise::run_shots_noisy_brute_with(
        |s| Box::new(crate::backend::stabilizer::StabilizerBackend::new(s)),
        &circuit,
        &noise,
        num_shots,
        42,
    )
    .unwrap();

    let traj_coh = trajectory.coherent_fraction();
    let brute_coh = brute.coherent_fraction();
    // 5 sigma for a pairwise comparison at 5000 shots.
    assert!(
        (traj_coh - brute_coh).abs() < 0.05,
        "coherent fraction: trajectory={traj_coh:.3}, brute={brute_coh:.3}"
    );
    assert!(traj_coh < 1.0, "noise should produce non-GHZ outcomes");

    for bit in 0..n {
        let t = trajectory.marginal(bit);
        let b = brute.marginal(bit);
        assert!(
            (t - b).abs() < 0.05,
            "bit {bit}: trajectory marginal {t:.3} vs brute {b:.3}"
        );
    }
}

#[test]
fn amplitude_damping_decays_to_ground() {
    // |1⟩ with strong amplitude damping should mostly decay to |0⟩
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]); // prepare |1⟩
    circuit.add_measure(0, 0);

    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.9);
    let factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::statevector::StatevectorBackend::new(s))
    };

    let result = run_trajectories(
        factory,
        &circuit,
        &noise,
        1000,
        42,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap();
    let num_zero = result.shots.iter().filter(|s| !s[0]).count();
    // With gamma=0.9 on |1⟩, P(decay) ≈ 0.9
    assert!(
        num_zero > 700,
        "strong AD should decay most to |0⟩, got {} zeros",
        num_zero
    );
}

#[test]
fn phase_damping_preserves_populations() {
    // Phase damping should not change |0⟩/|1⟩ populations
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::X, &[0]); // prepare |1⟩
    circuit.add_measure(0, 0);

    let mut pd_noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    for events in &mut pd_noise.after_gate {
        *events = events
            .iter()
            .map(|e| NoiseEvent {
                channel: NoiseChannel::PhaseDamping { gamma: 0.9 },
                qubits: e.qubits.clone(),
            })
            .collect();
    }

    let factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::statevector::StatevectorBackend::new(s))
    };

    let result = run_trajectories(
        factory,
        &circuit,
        &pd_noise,
        1000,
        42,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap();
    let num_one = result.shots.iter().filter(|s| s[0]).count();
    // Phase damping on |1⟩ should keep it as |1⟩ (only dephases superpositions)
    assert_eq!(num_one, 1000, "PD should not change |1⟩ population");
}

#[test]
fn readout_error_flips_bits() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_measure(0, 0); // measure |0⟩

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.with_readout_error(0.3, 0.0); // 30% chance of 0→1

    let factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::statevector::StatevectorBackend::new(s))
    };

    let result = run_trajectories(
        factory,
        &circuit,
        &noise,
        1000,
        42,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap();
    let num_one = result.shots.iter().filter(|s| s[0]).count();
    // Should see ~30% readout flips
    assert!(
        num_one > 200 && num_one < 400,
        "readout error p01=0.3 should flip ~30% of |0⟩ outcomes, got {}",
        num_one
    );
}

#[test]
fn two_qubit_depolarizing_produces_errors() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0] = vec![NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.5 },
        qubits: smallvec![0, 1],
    }];

    let factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::statevector::StatevectorBackend::new(s))
    };

    let result = run_trajectories(
        factory,
        &circuit,
        &noise,
        1000,
        42,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap();
    // With 50% 2q depolarizing, varied outcomes should appear.
    let num_00 = result.shots.iter().filter(|s| !s[0] && !s[1]).count();
    assert!(
        num_00 < 900,
        "strong 2q depolarizing should produce errors, got {} |00⟩",
        num_00
    );
}

#[test]
fn zero_noise_trajectory_deterministic() {
    let n = 3;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    let factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::statevector::StatevectorBackend::new(s))
    };

    let r1 = run_trajectories(
        factory,
        &circuit,
        &noise,
        100,
        42,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap();
    let r2 = run_trajectories(
        factory,
        &circuit,
        &noise,
        100,
        42,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap();
    assert_eq!(r1.shots, r2.shots, "same seed must produce same results");
}

fn grouped_circuit(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        circuit.add_gate(Gate::Ry(0.3 + 0.1 * q as f64), &[q]);
        circuit.add_gate(Gate::Rx(0.2 + 0.15 * q as f64), &[q]);
    }
    circuit.measure_all();
    circuit
}

fn statevector_route(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> ShotsResult {
    crate::sim::run_shots_with_noise(
        crate::sim::BackendKind::Statevector,
        circuit,
        noise,
        num_shots,
        seed,
        &ResolvedLimits::process_default(),
    )
    .unwrap()
}

fn per_shot(circuit: &Circuit, noise: &NoiseModel, num_shots: usize, seed: u64) -> ShotsResult {
    let factory = |s: u64| -> Box<dyn Backend> { Box::new(StatevectorBackend::new(s)) };
    run_trajectories(
        factory,
        circuit,
        noise,
        num_shots,
        seed,
        false,
        crate::sim::ResolvedBackend::Statevector,
    )
    .unwrap()
}

// Every branch kind the grouped path draws: symmetric depolarizing, an
// asymmetric Pauli weighted onto Y, the two-qubit pair channel, and
// asymmetric readout. The reference is the density-matrix distribution with
// the readout flips folded in bit by bit, checked per outcome and pooled as a
// chi-square at the Wilson-Hilferty 5-sigma quantile. The trailing rotations
// leave X, Y and Z errors after `T` distinguishable in the record; with `Ry`
// alone a Y and a Z there read the same, and swapping their rates passed.
#[test]
fn pauli_groups_sample_the_density_matrix_distribution_within_5_sigma() {
    let n = 5;
    let circuit = grouped_circuit(n);
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    for (index, instruction) in circuit.instructions.iter().enumerate() {
        if let Instruction::Gate { gate, targets } = instruction {
            if matches!(gate, Gate::Cx) {
                noise.after_gate[index].push(NoiseEvent {
                    channel: NoiseChannel::TwoQubitDepolarizing { p: 0.1 },
                    qubits: smallvec![targets[0], targets[1]],
                });
            }
            if matches!(gate, Gate::T) {
                noise.after_gate[index].push(NoiseEvent::pauli(targets[0], 0.005, 0.05, 0.01));
            }
        }
    }
    let quantum_only = noise.clone();
    let (p01, p10) = (0.02, 0.05);
    noise.with_readout_error(p01, p10);

    let num_shots = 100_000;
    let seed = 0xDEAD_BEEF;
    let groups = PauliGroups::sample(&circuit, &noise, num_shots, seed).unwrap();
    assert!(
        groups.num_groups() < num_shots / 2,
        "{} groups at {num_shots} shots leaves too little grouping to test",
        groups.num_groups()
    );
    let result = statevector_route(&circuit, &noise, num_shots, seed);
    assert_eq!(
        result.shots,
        run_pauli_groups(
            &groups,
            &circuit,
            &noise,
            seed,
            None,
            crate::sim::ResolvedBackend::Statevector,
            &ResolvedLimits::process_default()
        )
        .unwrap()
        .shots
    );

    let probs = crate::sim::noise::density_matrix_probabilities(
        &crate::sim::BackendKind::DensityMatrix,
        &circuit,
        &quantum_only,
        None,
        seed,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
    let dim = 1usize << n;
    let mut exact = vec![0.0f64; dim];
    for (x, &px) in probs.iter().enumerate() {
        for (y, slot) in exact.iter_mut().enumerate() {
            let mut weight = px;
            for bit in 0..n {
                weight *= match ((x >> bit) & 1, (y >> bit) & 1) {
                    (0, 0) => 1.0 - p01,
                    (0, _) => p01,
                    (_, 0) => p10,
                    _ => 1.0 - p10,
                };
            }
            *slot += weight;
        }
    }

    let mut counts = vec![0u64; dim];
    for shot in &result.shots {
        let index = shot
            .iter()
            .enumerate()
            .fold(0usize, |acc, (bit, &b)| acc | (usize::from(b) << bit));
        counts[index] += 1;
    }
    let total = num_shots as f64;
    for (index, (&count, &p)) in counts.iter().zip(&exact).enumerate() {
        let sigma = (p * (1.0 - p) / total).sqrt();
        let observed = count as f64 / total;
        assert!(
            (observed - p).abs() <= 5.0 * sigma,
            "outcome {index}: exact {p}, sampled {observed}, tolerance {}",
            5.0 * sigma
        );
    }

    let chi_square: f64 = counts
        .iter()
        .zip(&exact)
        .map(|(&count, &p)| {
            let expected = p * total;
            (count as f64 - expected).powi(2) / expected
        })
        .sum();
    let dof = (dim - 1) as f64;
    let spread = (2.0 / (9.0 * dof)).sqrt();
    let bound = dof * (1.0 - 2.0 / (9.0 * dof) + 5.0 * spread).powi(3);
    assert!(
        chi_square < bound,
        "chi-square {chi_square:.1} over {dof} degrees of freedom exceeds {bound:.1}"
    );
}

#[test]
fn tensor_network_groups_draw_the_statevector_records() {
    let circuit = grouped_circuit(6);
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
    noise.with_readout_error(0.03, 0.05);
    let (shots, seed) = (400, 42);
    let groups = PauliGroups::sample(&circuit, &noise, shots, seed).unwrap();
    assert!(groups.num_groups() < shots / 2);

    let network = crate::sim::run_shots_with_noise(
        crate::sim::BackendKind::TensorNetwork,
        &circuit,
        &noise,
        shots,
        seed,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
    assert_eq!(
        network.shots,
        statevector_route(&circuit, &noise, shots, seed).shots
    );
}

#[test]
fn pauli_groups_decline_what_draws_against_the_state_or_an_outcome() {
    let n = 4;
    let seed = 42;
    let shots = 200;
    let eligible = grouped_circuit(n);
    let low = |c: &Circuit| NoiseModel::uniform_depolarizing(c, 1e-3);
    assert!(PauliGroups::sample(&eligible, &low(&eligible), shots, seed).is_some());

    let mut readout_only = NoiseModel::uniform_depolarizing(&eligible, 0.0);
    readout_only.with_readout_error(0.1, 0.1);
    let groups = PauliGroups::sample(&eligible, &readout_only, shots, seed).unwrap();
    assert_eq!(groups.num_groups(), 1);

    let mut mid_measured = Circuit::new(n, n + 1);
    mid_measured.add_gate(Gate::H, &[0]);
    mid_measured.add_measure(0, n);
    mid_measured
        .instructions
        .extend(eligible.instructions.iter().cloned());
    let mut reset = eligible.clone();
    reset
        .instructions
        .insert(n, Instruction::Reset { qubit: 1 });
    let mut conditional = eligible.clone();
    conditional.instructions.insert(
        n,
        Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: smallvec![1],
        },
    );
    let unmeasured = eligible.without_measurements();
    for (label, circuit) in [
        ("mid-circuit measurement", &mid_measured),
        ("reset", &reset),
        ("conditional", &conditional),
        ("no measurement", &unmeasured),
    ] {
        assert!(
            PauliGroups::sample(circuit, &low(circuit), shots, seed).is_none(),
            "{label} must keep the per-shot path"
        );
    }

    let damping = NoiseModel::with_amplitude_damping(&eligible, 0.01);
    assert!(PauliGroups::sample(&eligible, &damping, shots, seed).is_none());

    let mut after_measure = low(&eligible);
    let first_measure = eligible
        .instructions
        .iter()
        .position(|i| matches!(i, Instruction::Measure { .. }))
        .unwrap();
    after_measure.after_gate[first_measure].push(NoiseEvent::pauli(1, 0.01, 0.0, 0.0));
    assert!(PauliGroups::sample(&eligible, &after_measure, shots, seed).is_none());
    after_measure.after_gate[first_measure][0] = NoiseEvent::pauli(1, 0.0, 0.0, 0.0);
    assert!(PauliGroups::sample(&eligible, &after_measure, shots, seed).is_some());

    let saturated = NoiseModel::uniform_depolarizing(&eligible, 0.5);
    assert!(
        PauliGroups::sample(&eligible, &saturated, shots, seed).is_none(),
        "near one pattern per shot must fall back"
    );
}

// The route check, end to end: a circuit the grouping declines, and an
// eligible circuit on a route other than the host statevector, both return
// the per-shot trajectories bit for bit.
#[test]
fn ineligible_noisy_shots_keep_the_per_shot_trajectories() {
    let n = 6;
    let seed = 42;
    let mut mid_measured = Circuit::new(n, n + 1);
    mid_measured.add_gate(Gate::H, &[0]);
    mid_measured.add_gate(Gate::T, &[0]);
    mid_measured.add_measure(0, n);
    mid_measured
        .instructions
        .extend(grouped_circuit(n).instructions);
    let noise = NoiseModel::uniform_depolarizing(&mid_measured, 1e-3);
    assert_eq!(
        statevector_route(&mid_measured, &noise, 300, seed).shots,
        per_shot(&mid_measured, &noise, 300, seed).shots
    );

    let mut product = Circuit::new(n, n);
    for q in 0..n {
        product.add_gate(Gate::H, &[q]);
        product.add_gate(Gate::T, &[q]);
    }
    product.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&product, 1e-3);
    assert!(PauliGroups::sample(&product, &noise, 300, seed).is_some());
    let auto = crate::sim::run_shots_with_noise(
        crate::sim::BackendKind::Auto,
        &product,
        &noise,
        300,
        seed,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
    assert_eq!(
        auto.metadata.backend,
        crate::sim::ResolvedBackend::ProductState
    );
    let factory = |s: u64| -> Box<dyn Backend> {
        Box::new(crate::backend::product::ProductStateBackend::new(s))
    };
    let product_per_shot = run_trajectories(
        factory,
        &product,
        &noise,
        300,
        seed,
        false,
        crate::sim::ResolvedBackend::ProductState,
    )
    .unwrap();
    assert_eq!(auto.shots, product_per_shot.shots);
}

// Shot `i` of a run seeded 42 must not reappear as shot `i + k` of a run
// seeded 43, which a seed of `seed + i` would give.
#[test]
fn adjacent_run_seeds_draw_unrelated_grouped_shots() {
    let n = 8;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 1e-3);
    let shots = 200;
    let groups = PauliGroups::sample(&circuit, &noise, shots, 42).unwrap();
    assert!(groups.num_groups() < shots / 2);

    let a = statevector_route(&circuit, &noise, shots, 42);
    let b = statevector_route(&circuit, &noise, shots, 43);
    for shift in 0..=4 {
        let aligned = (0..shots - shift)
            .filter(|&i| a.shots[i + shift] == b.shots[i] || a.shots[i] == b.shots[i + shift])
            .count();
        assert!(
            aligned < shots / 10,
            "runs seeded 42 and 43 agree at {aligned} positions under shift {shift}"
        );
    }
}

// From 10 qubits the fusion passes rewrite the prefix: 1q runs at 10, 2q
// blocks at 12, tiled runs at 14 and diagonal batches at 16. The error-free
// group runs that fused plan and must end on the raw stream's state, and its
// members must draw the records the raw state would give them.
#[test]
fn error_free_group_matches_the_raw_prefix_above_the_fusion_floors() {
    let seed = 42;
    let shots = 300;
    for n in [10usize, 12, 14, 16] {
        let mut circuit = Circuit::new(n, n);
        for layer in 0..4 {
            for q in 0..n {
                circuit.add_gate(Gate::H, &[q]);
                circuit.add_gate(Gate::T, &[q]);
                circuit.add_gate(Gate::Rz(0.03 * (layer + q + 1) as f64), &[q]);
            }
            for q in 0..n - 1 {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
                circuit.add_gate(Gate::Rzz(0.1 * (q + 1) as f64), &[q, q + 1]);
            }
        }
        circuit.measure_all();
        let noise = NoiseModel::uniform_depolarizing(&circuit, 1e-4);
        let groups = PauliGroups::sample(&circuit, &noise, shots, seed).unwrap();
        assert!(groups.num_groups() > 1, "{n} qubits: no error group drawn");
        let error_free = groups.patterns.iter().position(Vec::is_empty).unwrap();

        let prefix = circuit.with_instructions(circuit.instructions[..groups.prefix].to_vec());
        let fused_len = crate::sim::fuse_for_backend(&StatevectorBackend::new(seed), &prefix)
            .instructions
            .len();
        assert!(
            fused_len < groups.prefix,
            "{n} qubits: fusion kept {fused_len} of {} instructions",
            groups.prefix
        );

        let mut fused = StatevectorBackend::new(seed);
        evolve_error_free(&mut fused, &circuit, groups.prefix).unwrap();
        let mut raw = StatevectorBackend::new(seed);
        raw.init(n, n).unwrap();
        evolve_pattern(
            &mut raw,
            &groups,
            error_free,
            &circuit,
            &noise,
            0,
            groups.prefix,
            0,
        )
        .unwrap();
        let raw_probs = raw.probabilities().unwrap();
        let fused_probs = fused.probabilities().unwrap();
        for (index, (f, r)) in fused_probs.iter().zip(&raw_probs).enumerate() {
            assert!(
                (f - r).abs() < 1e-12,
                "{n} qubits, basis {index}: fused {f} vs raw {r}"
            );
        }

        let members = groups.members(error_free);
        let mut draws: Vec<(f64, usize)> = members
            .iter()
            .enumerate()
            .map(|(k, &shot)| {
                let mut rng = ChaCha8Rng::seed_from_u64(crate::sim::mix_seed(seed, shot));
                (rand::RngExt::random::<f64>(&mut rng), k)
            })
            .collect();
        draws.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let outcomes = inverse_cdf(&draws, raw_probs);
        let result = run_pauli_groups(
            &groups,
            &circuit,
            &noise,
            seed,
            None,
            crate::sim::ResolvedBackend::Statevector,
            &ResolvedLimits::process_default(),
        )
        .unwrap();
        for (&shot, &basis) in members.iter().zip(&outcomes) {
            let expected: Vec<bool> = (0..n).map(|q| (basis >> q) & 1 == 1).collect();
            assert_eq!(result.shots[shot], expected, "{n} qubits, shot {shot}");
        }
    }
}

// Each group evolved from |0...0⟩ on a backend of its own, the path the
// shared checkpoint replaces. Its records are what the checkpointed runs
// must reproduce bit for bit.
fn records_from_scratch(
    groups: &PauliGroups,
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
) -> Vec<Vec<bool>> {
    let meas_map = circuit.measurement_map();
    let readout = written_readout(circuit, &noise.readout);
    let mut shots = vec![Vec::new(); groups.members.len()];
    for group in 0..groups.num_groups() {
        let mut backend =
            StatevectorBackend::new(crate::sim::mix_seed(seed, groups.members(group)[0]));
        if groups.patterns[group].is_empty() {
            evolve_error_free(&mut backend, circuit, groups.prefix).unwrap();
        } else {
            backend
                .init(circuit.num_qubits, circuit.num_classical_bits)
                .unwrap();
            evolve_pattern(
                &mut backend,
                groups,
                group,
                circuit,
                noise,
                0,
                groups.prefix,
                0,
            )
            .unwrap();
        }
        let scale = backend.probability_scale();
        let weights = backend.state_vector().iter().map(|a| a.norm_sqr() * scale);
        let records = group_records(groups, group, circuit, &meas_map, &readout, seed, weights);
        for (&shot, record) in groups.members(group).iter().zip(records) {
            shots[shot] = record;
        }
    }
    shots
}

fn layered_noise_circuit(n: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n, n);
    for layer in 0..depth {
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
            circuit.add_gate(Gate::T, &[q]);
            circuit.add_gate(Gate::Rz(0.03 * (layer + q + 1) as f64), &[q]);
        }
        for q in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    circuit.measure_all();
    circuit
}

#[test]
fn checkpointed_groups_draw_the_records_of_groups_run_from_scratch() {
    let seed = 42;
    let shots = 400;
    for n in [6usize, 10, 12] {
        let circuit = layered_noise_circuit(n, 2);
        for p in [1e-3, 1e-2] {
            let mut noise = NoiseModel::uniform_depolarizing(&circuit, p);
            noise.with_readout_error(0.02, 0.03);
            let groups = PauliGroups::sample(&circuit, &noise, shots, seed).unwrap();
            assert!(groups.num_groups() > 2, "{n} qubits, p {p}: too few groups");
            let reference = records_from_scratch(&groups, &circuit, &noise, seed);

            let grouped = run_pauli_groups(
                &groups,
                &circuit,
                &noise,
                seed,
                None,
                crate::sim::ResolvedBackend::Statevector,
                &ResolvedLimits::process_default(),
            )
            .unwrap();
            assert_eq!(grouped.shots, reference, "{n} qubits, p {p}");

            let mut order: Vec<usize> = (0..groups.num_groups()).collect();
            order.sort_by_key(|&group| groups.starts[group]);
            let meas_map = circuit.measurement_map();
            let readout = written_readout(&circuit, &noise.readout);
            let run = |chunk: &[usize]| {
                run_statevector_chunk(
                    &groups,
                    chunk,
                    &circuit,
                    &noise,
                    &meas_map,
                    &readout,
                    seed,
                    &ResolvedLimits::process_default(),
                )
                .unwrap()
            };
            let whole = collect_groups(
                &groups,
                &circuit,
                crate::sim::ResolvedBackend::Statevector,
                run(&order),
            )
            .unwrap();
            assert_eq!(whole.shots, reference, "{n} qubits, p {p}, one chunk");

            #[cfg(feature = "parallel")]
            for parts in [2, 3, 7] {
                let chunks = work_chunks(&groups, &order, parts);
                assert!(chunks.len() <= parts);
                assert_eq!(chunks.concat(), order);
                let split = collect_groups(
                    &groups,
                    &circuit,
                    crate::sim::ResolvedBackend::Statevector,
                    chunks.into_iter().flat_map(run),
                )
                .unwrap();
                assert_eq!(split.shots, reference, "{n} qubits, p {p}, {parts} chunks");
            }
        }
    }
}

// Grouping moves no draw of the error stream, so a shot fires the errors it
// fired on the per-shot path. With the outcome fixed by the circuit, the
// records then agree shot for shot.
#[test]
fn grouped_shots_fire_the_per_shot_errors() {
    let n = 3;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.measure_all();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0].push(NoiseEvent::pauli(0, 0.05, 0.0, 0.0));
    noise.after_gate[n].push(NoiseEvent {
        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.1 },
        qubits: smallvec![1, 2],
    });
    let shots = 500;
    assert!(PauliGroups::sample(&circuit, &noise, shots, 7).is_some());
    let grouped = statevector_route(&circuit, &noise, shots, 7);
    let reference = per_shot(&circuit, &noise, shots, 7);
    assert_eq!(grouped.shots, reference.shots);
    assert!(grouped.shots.iter().any(|s| s.iter().any(|&b| b)));
}

// Layers, a measurement of qubit 0 that conditions an X on qubit 1, a reset,
// more layers with live noise, then every qubit measured with readout error.
// Each shot finishes its own trajectory from its group's state, so the
// records match the per-shot trajectories bit for bit, whole and split.
#[test]
fn replayed_groups_keep_the_per_shot_trajectories() {
    let seed = 42;
    let shots = 300;
    for n in [6usize, 10] {
        let mut circuit = Circuit::new(n, n + 1);
        let layer = |circuit: &mut Circuit, offset: usize| {
            for q in 0..n {
                circuit.add_gate(Gate::H, &[q]);
                circuit.add_gate(Gate::T, &[q]);
                circuit.add_gate(Gate::Rz(0.03 * (offset + q + 1) as f64), &[q]);
            }
            for q in 0..n - 1 {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        };
        layer(&mut circuit, 0);
        layer(&mut circuit, 1);
        circuit.add_measure(0, n);
        circuit.instructions.push(Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(n),
            gate: Gate::X,
            targets: smallvec![1],
        });
        circuit.add_reset(0);
        layer(&mut circuit, 2);
        circuit.measure_all();
        for p in [1e-3, 1e-2] {
            let mut noise = NoiseModel::uniform_depolarizing(&circuit, p);
            noise.after_gate[n].push(NoiseEvent {
                channel: NoiseChannel::TwoQubitDepolarizing { p: 0.05 },
                qubits: smallvec![0, 1],
            });
            noise.with_readout_error(0.02, 0.03);
            let groups = PauliGroups::sample_replayed(&circuit, &noise, shots, seed).unwrap();
            assert!(groups.num_groups() > 1, "{n} qubits, p {p}: one group");
            assert!(groups.num_groups() < shots, "{n} qubits, p {p}: no sharing");
            let reference = per_shot(&circuit, &noise, shots, seed);
            let routed = statevector_route(&circuit, &noise, shots, seed);
            assert_eq!(routed.shots, reference.shots, "{n} qubits, p {p}");

            let mut order: Vec<usize> = (0..groups.num_groups()).collect();
            order.sort_by_key(|&group| groups.starts[group]);
            let readout = written_readout(&circuit, &noise.readout);
            let run = |segments: Vec<Segment>| {
                run_replayed_chunk(
                    &groups,
                    &segments,
                    &circuit,
                    &noise,
                    &readout,
                    seed,
                    &ResolvedLimits::process_default(),
                )
                .unwrap()
            };
            #[cfg(feature = "parallel")]
            for parts in [2, 3, 7] {
                let chunks = replay_chunks(&groups, &order, parts, 5);
                assert!(chunks.len() <= parts);
                let split = fold_replayed(
                    &circuit,
                    crate::sim::ResolvedBackend::Statevector,
                    shots,
                    chunks.into_iter().flat_map(run),
                )
                .unwrap();
                assert_eq!(
                    split.shots, reference.shots,
                    "{n} qubits, p {p}, {parts} chunks"
                );
            }
            let whole = order
                .iter()
                .map(|&group| (group, 0..groups.members(group).len()))
                .collect();
            let serial = fold_replayed(
                &circuit,
                crate::sim::ResolvedBackend::Statevector,
                shots,
                run(whole),
            )
            .unwrap();
            assert_eq!(serial.shots, reference.shots, "{n} qubits, p {p}, serial");
        }
    }
    let terminal = grouped_circuit(4);
    let noise = NoiseModel::uniform_depolarizing(&terminal, 1e-3);
    assert!(PauliGroups::sample_replayed(&terminal, &noise, shots, seed).is_none());
    let damping = NoiseModel::with_amplitude_damping(&terminal, 0.01);
    assert!(PauliGroups::sample_replayed(&terminal, &damping, shots, seed).is_none());
}

fn every_channel(memory: bool) -> Vec<NoiseChannel> {
    let r = |x: f64| Complex64::new(x, 0.0);
    let zero = r(0.0);
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let (c, s) = (0.7f64.sqrt(), 0.3f64.sqrt());
    let thermal = |t1, t2, gate_time, excited_population| NoiseChannel::ThermalRelaxation {
        t1,
        t2,
        gate_time,
        excited_population,
    };
    // Damping in a Hadamard-rotated frame: effects with off-diagonal terms,
    // so branch weights read the coherence.
    let custom = vec![
        [[r(half), r(half)], [r(c * half), r(-c * half)]],
        [[r(s * half), r(-s * half)], [zero, zero]],
    ];
    let mut a0 = [[zero; 4]; 4];
    for (t, row) in a0.iter_mut().enumerate().take(3) {
        row[t] = r(1.0);
    }
    a0[3][3] = r(c);
    let mut a1 = [[zero; 4]; 4];
    a1[0][3] = r(s);
    let times_h_on_high = |a: [[Complex64; 4]; 4]| {
        let h = [[half, half], [half, -half]];
        let mut k = [[zero; 4]; 4];
        for (t, row) in k.iter_mut().enumerate() {
            for (u, entry) in row.iter_mut().enumerate() {
                for m in (u & 1..4).step_by(2) {
                    *entry += a[t][m] * h[m >> 1][u >> 1];
                }
            }
        }
        k
    };
    let mut channels = vec![
        NoiseChannel::Pauli {
            px: 0.02,
            py: 0.05,
            pz: 0.03,
        },
        NoiseChannel::Depolarizing { p: 0.12 },
        NoiseChannel::AmplitudeDamping { gamma: 0.2 },
        NoiseChannel::PhaseDamping { gamma: 0.15 },
        thermal(40.0, 55.0, 6.0, 0.0),
        thermal(30.0, 50.0, 4.0, 0.25),
        thermal(30.0, 50.0, 0.0, 0.25),
        NoiseChannel::TwoQubitDepolarizing { p: 0.2 },
        NoiseChannel::Custom { kraus: custom },
        NoiseChannel::Kraus2q {
            kraus: vec![times_h_on_high(a0), times_h_on_high(a1)],
        },
    ];
    if memory {
        channels.extend([
            NoiseChannel::Leakage { p: 0.1 },
            NoiseChannel::Seepage { p: 0.3 },
            NoiseChannel::LeakageTransport { p: 0.4 },
            NoiseChannel::QuasiStatic {
                axis: PauliAxis::Z,
                weights: vec![(0, 0.3), (1, 0.1)],
            },
            NoiseChannel::QuasiStatic {
                axis: PauliAxis::X,
                weights: vec![(1, 0.2)],
            },
        ]);
    }
    channels
}

fn pauli_channels(scale: f64) -> Vec<NoiseChannel> {
    vec![
        NoiseChannel::Pauli {
            px: 0.02 * scale,
            py: 0.05 * scale,
            pz: 0.03 * scale,
        },
        NoiseChannel::Depolarizing { p: 0.12 * scale },
        NoiseChannel::TwoQubitDepolarizing { p: 0.2 * scale },
    ]
}

fn cycled_model(circuit: &Circuit, slots: usize, channels: &[NoiseChannel]) -> NoiseModel {
    let n = circuit.num_qubits;
    let mut after_gate = vec![Vec::new(); slots];
    let mut next = 0usize;
    for (slot, events) in after_gate.iter_mut().enumerate() {
        for _ in 0..2 {
            let channel = channels[next % channels.len()].clone();
            let q = (slot + next) % n;
            let qubits = if channel.num_qubits() == 2 {
                smallvec![q, (q + 1) % n]
            } else {
                smallvec![q]
            };
            events.push(NoiseEvent { channel, qubits });
            next += 1;
        }
    }
    let mut readout = vec![None; circuit.num_classical_bits];
    readout[1] = Some(ReadoutError {
        p01: 0.05,
        p10: 0.08,
    });
    NoiseModel {
        after_gate,
        readout,
    }
}

fn push_region(circuit: &mut Circuit, body: Vec<Instruction>) {
    let condition = crate::circuit::ClassicalCondition::BitIsOne(0);
    circuit
        .instructions
        .push(crate::circuit::guarded(condition, body).unwrap());
}

fn one_gate(gate: Gate, qubit: usize) -> Instruction {
    Instruction::Gate {
        gate,
        targets: smallvec![qubit],
    }
}

fn mixed_circuit(regions: bool) -> Circuit {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Ry(0.73), &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Rx(0.41), &[2]);
    circuit.add_gate(Gate::Cz, &[1, 2]);
    circuit.add_measure(0, 0);
    if regions {
        push_region(
            &mut circuit,
            vec![one_gate(Gate::X, 1), one_gate(Gate::Ry(0.2), 2)],
        );
    }
    circuit.add_gate(Gate::Cx, &[2, 0]);
    circuit.add_gate(Gate::Ry(1.1), &[0]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);
    circuit
}

fn clifford_circuit(regions: bool) -> Circuit {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::S, &[2]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::H, &[1]);
    if regions {
        push_region(
            &mut circuit,
            vec![one_gate(Gate::X, 2), one_gate(Gate::H, 1)],
        );
    }
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);
    circuit
}

fn fingerprint(result: &ShotsResult) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for bits in result.shots.iter().chain(result.leaked.iter().flatten()) {
        for &bit in bits {
            hash = (hash ^ u64::from(bit)).wrapping_mul(0x0100_0000_01b3);
        }
        hash = (hash ^ 2).wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

// Golden records from the per-event channel match, on every trajectory loop:
// flat and guarded slot layouts, with and without per-shot memory, on each
// backend that runs dense Kraus channels, then the Pauli frame replay and
// both grouped Pauli routes.
#[test]
fn every_channel_keeps_its_seeded_records() {
    use crate::backend::factored::FactoredBackend;
    use crate::backend::mps::MpsBackend;
    use crate::backend::sparse::SparseBackend;
    use crate::sim::ResolvedBackend;

    type Factory = fn(u64) -> Box<dyn Backend>;
    let backends: [(Factory, ResolvedBackend); 4] = [
        (
            |s| Box::new(StatevectorBackend::new(s)),
            ResolvedBackend::Statevector,
        ),
        (|s| Box::new(MpsBackend::new(s, 16)), ResolvedBackend::Mps),
        (
            |s| Box::new(FactoredBackend::new(s)),
            ResolvedBackend::Factored,
        ),
        (|s| Box::new(SparseBackend::new(s)), ResolvedBackend::Sparse),
    ];
    let shots = 96;
    let seed = 0xDEAD_BEEF;
    let slots = |circuit: &Circuit, regions: bool| {
        if regions {
            circuit.noise_slot_count()
        } else {
            circuit.instructions.len()
        }
    };
    let mut prints = Vec::new();
    for regions in [false, true] {
        let circuit = mixed_circuit(regions);
        for memory in [false, true] {
            let noise = cycled_model(&circuit, slots(&circuit, regions), &every_channel(memory));
            noise.validate_for(&circuit).unwrap();
            for (factory, route) in backends {
                let run = |force_serial| {
                    run_trajectories(factory, &circuit, &noise, shots, seed, force_serial, route)
                        .unwrap()
                };
                let serial = run(true);
                assert_eq!(run(false).shots, serial.shots);
                prints.push(fingerprint(&serial));
            }
        }
    }

    for regions in [false, true] {
        let circuit = clifford_circuit(regions);
        let noise = cycled_model(&circuit, slots(&circuit, regions), &pauli_channels(1.0));
        noise.validate_for(&circuit).unwrap();
        let brute = crate::sim::noise::run_shots_noisy_brute_with(
            |s| Box::new(crate::backend::stabilizer::StabilizerBackend::new(s)),
            &circuit,
            &noise,
            shots,
            seed,
        )
        .unwrap();
        prints.push(fingerprint(&brute));
    }

    let limits = ResolvedLimits::process_default();
    let terminal = grouped_circuit(4);
    let mut noise = cycled_model(&terminal, terminal.instructions.len(), &pauli_channels(0.1));
    let first_measure = terminal.num_qubits * 4 + terminal.num_qubits - 1;
    for events in &mut noise.after_gate[first_measure..] {
        events.clear();
    }
    let groups = PauliGroups::sample(&terminal, &noise, shots, seed).unwrap();
    let grouped = run_pauli_groups(
        &groups,
        &terminal,
        &noise,
        seed,
        None,
        ResolvedBackend::Statevector,
        &limits,
    )
    .unwrap();
    prints.push(fingerprint(&grouped));

    let mid = clifford_circuit(false);
    let noise = cycled_model(&mid, mid.instructions.len(), &pauli_channels(0.1));
    let groups = PauliGroups::sample_replayed(&mid, &noise, shots, seed).unwrap();
    let replayed = run_replayed_groups(
        &groups,
        &mid,
        &noise,
        seed,
        ResolvedBackend::Statevector,
        &limits,
    )
    .unwrap();
    prints.push(fingerprint(&replayed));

    let [flat, flat_memory, guarded, guarded_memory] = [
        2165866839194561914,
        2435124084252643758,
        2039674417054972793,
        16645300680079721348,
    ];
    let mut expected = Vec::new();
    for print in [flat, flat_memory, guarded, guarded_memory] {
        expected.extend([print; 4]);
    }
    expected.extend([
        11151965011361772201,
        448741705247321696,
        4716965702601468798,
        3930197763577050471,
    ]);
    assert_eq!(prints, expected);
}

fn copy_of(backend: &StatevectorBackend) -> StatevectorBackend {
    let mut copy = StatevectorBackend::new(7);
    copy.copy_state_from(backend);
    copy
}

#[track_caller]
fn assert_states_close(got: &StatevectorBackend, want: &StatevectorBackend, label: &str) {
    let (got, want) = (
        got.export_statevector().unwrap(),
        want.export_statevector().unwrap(),
    );
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).norm() < 1e-12, "{label}: amplitude {i} {g} vs {w}");
    }
}

/// A seeded five-qubit state, and the same state after measuring qubit 4, which
/// leaves a deferred norm and a qubit whose `p1` is zero or one.
fn damping_states() -> [StatevectorBackend; 2] {
    use rand::RngExt;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let amplitudes: Vec<Complex64> = (0..32)
        .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect();
    let norm = amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    let mut fresh = StatevectorBackend::new(42);
    fresh
        .init_from_state(amplitudes.iter().map(|a| a / norm).collect(), 1)
        .unwrap();
    let mut measured = copy_of(&fresh);
    measured
        .apply(&Instruction::Measure {
            qubit: 4,
            classical_bit: 0,
        })
        .unwrap();
    assert_ne!(measured.pending_norm, 1.0);
    [fresh, measured]
}

#[test]
fn one_pass_damping_matches_the_two_step_branch() {
    use crate::backend::sealed::Sealed;
    for (index, state) in damping_states().iter().enumerate() {
        for gamma in [1e-3, 0.3] {
            let damping = Damping::new(gamma);
            for qubit in 0..5 {
                let read = copy_of(state)
                    .damp_no_jump(qubit, gamma, damping.keep)
                    .unwrap();
                assert_eq!(read, state.qubit_probability(qubit).unwrap());
                for moves in [true, false] {
                    for r in [0.0, 0.5 * gamma, gamma, 0.5, 1.0 - f64::EPSILON] {
                        let (mut fused, mut two_step) = (copy_of(state), copy_of(state));
                        apply_diagonal_kraus_2op(&mut fused, qubit, damping, moves, r).unwrap();
                        apply_damping_two_step(&mut two_step, qubit, damping, moves, r).unwrap();
                        let label = format!("state {index} gamma {gamma} q{qubit} {moves} r {r}");
                        assert_states_close(&fused, &two_step, &label);
                    }
                }
            }
        }
    }
}

// `|1>` under a deferred norm a few ulps above one reads `p1 > 1`, so `r = gamma`
// lies under `gamma * p1` and the jump wins after the pass has applied the no-jump
// branch. At `gamma` just under one the branch norm `1 - gamma` sits at the floor,
// which leaves the state alone.
#[test]
fn one_pass_damping_keeps_the_rule_at_its_edges() {
    let mut excited = StatevectorBackend::new(42);
    excited.init(1, 0).unwrap();
    excited.apply(&one_gate(Gate::X, 0)).unwrap();
    let mut rounded = copy_of(&excited);
    rounded.pending_norm = 1.0 + 4.0 * f64::EPSILON;
    assert!(0.5 < 0.5 * rounded.qubit_probability(0).unwrap());

    let floor = 1.0 - 1e-13;
    for (state, gamma, r, label) in [
        (&rounded, 0.5, 0.5, "rounded jump"),
        (&excited, floor, 1.0 - 1e-14, "norm at the floor"),
    ] {
        for moves in [true, false] {
            let damping = Damping::new(gamma);
            let (mut fused, mut two_step) = (copy_of(state), copy_of(state));
            apply_diagonal_kraus_2op(&mut fused, 0, damping, moves, r).unwrap();
            apply_damping_two_step(&mut two_step, 0, damping, moves, r).unwrap();
            assert_states_close(&fused, &two_step, &format!("{label} {moves}"));
        }
    }
    let mut decayed = copy_of(&rounded);
    apply_diagonal_kraus_2op(&mut decayed, 0, Damping::new(0.5), true, 0.5).unwrap();
    assert!((decayed.qubit_probability(0).unwrap()).abs() < 1e-12);
}

#[test]
fn a_no_jump_damping_event_is_one_pass() {
    use crate::backend::statevector::sweeps::take_ticks;
    let [state, _] = damping_states();
    let damping = Damping::new(1e-3);
    let (mut fused, mut two_step) = (copy_of(&state), copy_of(&state));
    take_ticks();
    apply_diagonal_kraus_2op(&mut fused, 2, damping, true, 0.5).unwrap();
    assert_eq!(take_ticks(), 1);
    apply_damping_two_step(&mut two_step, 2, damping, true, 0.5).unwrap();
    assert_eq!(take_ticks(), 2);
}

// A qubit held at `|1>`: nine no-jump branches, then a jump that renormalizes and an
// `X` that re-excites. Each no-jump branch halves the raw `|1>` weight and doubles
// the squared deferred norm, so the 1800 here would carry both past the range of an
// `f64` without the fold.
#[test]
fn a_long_trajectory_keeps_its_deferred_norm_in_range() {
    let mut backend = StatevectorBackend::new(42);
    backend.init(1, 0).unwrap();
    backend.apply(&one_gate(Gate::X, 0)).unwrap();
    let damping = Damping::new(0.5);
    for step in 0..2000 {
        let jump = step % 10 == 9;
        let r = if jump { 0.0 } else { 0.9 };
        apply_diagonal_kraus_2op(&mut backend, 0, damping, true, r).unwrap();
        if jump {
            backend.apply(&one_gate(Gate::X, 0)).unwrap();
        }
    }
    assert!((backend.qubit_probability(0).unwrap() - 1.0).abs() < 1e-9);
    let norm: f64 = backend
        .export_statevector()
        .unwrap()
        .iter()
        .map(|a| a.norm_sqr())
        .sum();
    assert!((norm - 1.0).abs() < 1e-9, "norm {norm}");
}

fn damping_ansatz(n: usize, layers: usize) -> Circuit {
    let mut circuit = circuits::hardware_efficient_ansatz(n, layers, 42);
    circuit.measure_all();
    circuit
}

/// Amplitude damping on every gate target, phase damping when `dephasing`.
fn damping_model(circuit: &Circuit, gamma: f64, dephasing: bool) -> NoiseModel {
    let mut noise = NoiseModel::with_amplitude_damping(circuit, gamma);
    if dephasing {
        for event in noise.after_gate.iter_mut().flatten() {
            event.channel = NoiseChannel::PhaseDamping { gamma };
        }
    }
    noise
}

fn shared_prefix(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Option<ShotsResult> {
    run_shared_prefix(
        circuit,
        noise,
        num_shots,
        seed,
        crate::sim::ResolvedBackend::Statevector,
        &ResolvedLimits::process_default(),
    )
    .unwrap()
}

#[track_caller]
fn assert_same_records(got: &ShotsResult, want: &ShotsResult, label: &str) {
    assert_eq!(got.shots, want.shots, "{label}");
    assert_eq!(got.counts(), want.counts(), "{label}");
    assert_eq!(
        format!("{:?}", got.metadata),
        format!("{:?}", want.metadata),
        "{label}"
    );
}

// Rates from one where most shots never jump to one where every shot leaves the
// shared state on its first event, and shot counts from none up past the split.
#[test]
fn shared_prefix_keeps_the_per_shot_records() {
    let circuit = damping_ansatz(6, 2);
    for seed in [42, 7, 0xDEAD_BEEF] {
        for gamma in [1e-3, 0.05, 0.6, 1.0] {
            for dephasing in [false, true] {
                let noise = damping_model(&circuit, gamma, dephasing);
                for shots in [0, 1, 3, 4, 150] {
                    let label = format!("seed {seed} gamma {gamma} {dephasing} {shots} shots");
                    let shared = shared_prefix(&circuit, &noise, shots, seed).expect(&label);
                    let reference = per_shot(&circuit, &noise, shots, seed);
                    assert_same_records(&shared, &reference, &label);
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn shared_prefix_records_do_not_depend_on_the_pool() {
    let circuit = damping_ansatz(10, 2);
    for (gamma, dephasing) in [(2e-3, false), (0.02, true)] {
        let noise = damping_model(&circuit, gamma, dephasing);
        let reference = per_shot(&circuit, &noise, 300, 42);
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let shared = pool
                .install(|| shared_prefix(&circuit, &noise, 300, 42))
                .unwrap();
            let label = format!("gamma {gamma}, {threads} threads");
            assert_same_records(&shared, &reference, &label);
        }
    }
}

// From 15 qubits the kernels split instead of the shots, so one walk carries every
// shot, those that leave and those that reach the end. One thread keeps the
// kernels' reductions in a fixed order on both sides.
#[cfg(feature = "parallel")]
#[test]
fn a_wide_shared_prefix_keeps_the_per_shot_records() {
    let circuit = damping_ansatz(16, 1);
    let noise = damping_model(&circuit, 0.01, false);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let shared = shared_prefix(&circuit, &noise, 24, 42).unwrap();
        assert_same_records(&shared, &per_shot(&circuit, &noise, 24, 42), "16 qubits");
    });
}

fn thermal(gate_time: f64) -> NoiseChannel {
    NoiseChannel::ThermalRelaxation {
        t1: 30.0,
        t2: 50.0,
        gate_time,
        excited_population: 0.1,
    }
}

// Damping, Pauli and pair channels and a thermal event over no time all share the
// gates. A thermal event that draws against the state ends the prefix, as does the
// first measurement, and the region, condition and reset past it run per shot. The
// statevector route takes the shared prefix and keeps the same records.
#[test]
fn shared_prefix_stops_at_the_first_unshared_instruction() {
    let shareable = [
        NoiseChannel::AmplitudeDamping { gamma: 0.08 },
        NoiseChannel::Depolarizing { p: 0.05 },
        NoiseChannel::PhaseDamping { gamma: 0.1 },
        NoiseChannel::TwoQubitDepolarizing { p: 0.06 },
        thermal(0.0),
    ];
    let mut ending = shareable.to_vec();
    ending.insert(3, thermal(4.0));

    let mut conditioned = Circuit::new(4, 5);
    conditioned
        .instructions
        .extend(circuits::hardware_efficient_ansatz(4, 1, 7).instructions);
    conditioned.add_measure(0, 4);
    conditioned.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(4),
        gate: Gate::X,
        targets: smallvec![1],
    });
    conditioned.add_reset(0);
    conditioned
        .instructions
        .extend(circuits::hardware_efficient_ansatz(4, 1, 8).instructions);
    conditioned.measure_all();

    let (shots, seed) = (200, 0xDEAD_BEEF);
    let circuits = [
        (mixed_circuit(false), false),
        (mixed_circuit(true), true),
        (conditioned, false),
    ];
    for (index, (circuit, regions)) in circuits.iter().enumerate() {
        let slots = if *regions {
            circuit.noise_slot_count()
        } else {
            circuit.instructions.len()
        };
        for channels in [&shareable[..], &ending[..]] {
            let noise = cycled_model(circuit, slots, channels);
            noise.validate_for(circuit).unwrap();
            let label = format!("circuit {index}, {} channels", channels.len());
            let reference = per_shot(circuit, &noise, shots, seed);
            let shared = shared_prefix(circuit, &noise, shots, seed).expect(&label);
            assert_same_records(&shared, &reference, &label);
            let routed = statevector_route(circuit, &noise, shots, seed);
            assert_eq!(routed.shots, reference.shots, "{label}");
        }
    }
}

// Per-shot memory, a model with no damping, damping only past an event that draws
// against the state or past the first measurement, and a state whose P(1) reads past
// the headroom all leave the per-shot path in charge.
#[test]
fn shared_prefix_declines_what_it_cannot_share() {
    let circuit = damping_ansatz(5, 2);
    let (shots, seed) = (60, 42);
    let event = |channel, qubit| NoiseEvent {
        channel,
        qubits: smallvec![qubit],
    };

    let mut leaky = damping_model(&circuit, 0.01, false);
    leaky.after_gate[0].push(event(NoiseChannel::Leakage { p: 0.05 }, 0));
    // The idle thermal event keeps the model off the grouped Pauli routes.
    let mut undamped = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    undamped.after_gate[0].push(event(thermal(0.0), 0));
    let mut drawing_first = damping_model(&circuit, 0.01, false);
    drawing_first.after_gate[0] = vec![event(thermal(4.0), 0)];
    let mut measured_first = Circuit::new(5, 6);
    measured_first.add_measure(0, 5);
    measured_first
        .instructions
        .extend(circuit.instructions.iter().cloned());

    // `diag(1, 1.01)` on `|1>` lifts P(1) to 1.0201 ahead of the second event.
    let mut gain = Circuit::new(2, 2);
    gain.add_gate(Gate::X, &[0]);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    gain.add_gate(
        Gate::Fused(Box::new([[one, zero], [zero, Complex64::new(1.01, 0.0)]])),
        &[0],
    );
    gain.add_gate(Gate::H, &[1]);
    gain.measure_all();

    let cases = [
        (&circuit, leaky, "per-shot memory"),
        (&circuit, undamped, "no damping"),
        (&circuit, drawing_first, "a drawing thermal event first"),
        (
            &measured_first,
            damping_model(&measured_first, 0.01, false),
            "a measurement first",
        ),
        (&gain, damping_model(&gain, 1e-3, false), "P(1) past one"),
    ];
    for (circuit, noise, label) in cases {
        assert!(
            shared_prefix(circuit, &noise, shots, seed).is_none(),
            "{label}"
        );
        assert_eq!(
            statevector_route(circuit, &noise, shots, seed).shots,
            per_shot(circuit, &noise, shots, seed).shots,
            "{label}"
        );
    }
}
