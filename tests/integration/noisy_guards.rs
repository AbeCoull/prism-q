//! Noise inside guarded bodies: the slot layout, the shots a body event fires
//! on, agreement with a deferred-measurement lowering, and the routes that
//! decline it.

use num_complex::Complex64;
use prism_q::circuit::{Circuit, ClassicalCondition, GuardedRegion, Instruction, guarded};
use prism_q::error::PrismError;
use prism_q::gates::Gate;
use prism_q::sim::BackendKind;
use prism_q::{
    GateFilter, NoiseBuilder, NoiseChannel, NoiseEvent, NoiseModel, compile_noisy, run_shots_noisy,
    simulate,
};
use smallvec::smallvec;

const SEED: u64 = 42;

fn gate(gate: Gate, targets: &[usize]) -> Instruction {
    Instruction::Gate {
        gate,
        targets: targets.iter().copied().collect(),
    }
}

fn region(condition: ClassicalCondition, body: Vec<Instruction>) -> Instruction {
    Instruction::Region(Box::new(GuardedRegion::new(condition, body)))
}

fn depolarizing(qubit: usize, p: f64) -> NoiseEvent {
    NoiseEvent::pauli(qubit, p / 3.0, p / 3.0, p / 3.0)
}

/// Histogram of each shot's classical record, bit `i` weighted `1 << i`.
fn histogram(shots: &[Vec<bool>], bits: usize) -> Vec<f64> {
    let mut counts = vec![0.0; 1 << bits];
    for shot in shots {
        let index = shot
            .iter()
            .enumerate()
            .fold(0, |acc, (bit, &set)| acc | (usize::from(set) << bit));
        counts[index] += 1.0;
    }
    counts.iter().map(|c| c / shots.len() as f64).collect()
}

fn assert_within_5_sigma(sampled: &[f64], exact: &[f64], shots: usize, label: &str) {
    for (outcome, (&f, &p)) in sampled.iter().zip(exact).enumerate() {
        let sigma = (p * (1.0 - p) / shots as f64).sqrt();
        assert!(
            (f - p).abs() <= 5.0 * sigma + 1e-3,
            "{label}: outcome {outcome:#b} sampled {f}, exact {p}"
        );
    }
}

#[test]
fn guarded_bodies_append_their_slots_after_the_top_level() {
    let mut circuit = Circuit::new(3, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(region(
        ClassicalCondition::BitIsOne(0),
        vec![
            gate(Gate::X, &[1]),
            Instruction::Conditional {
                condition: ClassicalCondition::BitIsZero(1),
                gate: Gate::Z,
                targets: smallvec![2],
            },
        ],
    ));
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::Y,
        targets: smallvec![2],
    });

    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.03);
    // Four top-level slots, the region's two body slots and its conditional's
    // gate slot, then the top-level conditional's gate slot.
    assert_eq!(noise.after_gate.len(), 8);
    let targets: Vec<Vec<usize>> = noise
        .after_gate
        .iter()
        .map(|slot| slot.iter().map(|event| event.qubits[0]).collect())
        .collect();
    assert_eq!(
        targets,
        vec![
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![1],
            vec![],
            vec![2],
            vec![2]
        ]
    );
    noise.validate_for(&circuit).expect("the layout validates");
}

// `h a; measure a; if (a) { x d; x d; }` with depolarizing after each body gate:
// the body is the identity, so `d` reads 1 only through noise, and only on the
// half of the shots that take the branch. A top-level event after the region
// fires on every shot.
#[test]
fn a_body_event_fires_only_on_the_shots_that_take_the_branch() {
    let p = 0.3;
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(region(
        ClassicalCondition::BitIsOne(0),
        vec![gate(Gate::X, &[1]), gate(Gate::X, &[1])],
    ));
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);

    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[0].clear();
    noise.after_gate[2] = vec![NoiseEvent::pauli(2, 0.25, 0.0, 0.0)];
    noise.after_gate[5] = vec![depolarizing(1, p)];
    noise.after_gate[6] = vec![depolarizing(1, p)];

    let flip = 2.0 * p / 3.0;
    let odd = 2.0 * flip * (1.0 - flip);
    let shots = 20_000;
    for (label, kind) in [
        ("auto", BackendKind::Auto),
        ("statevector", BackendKind::Statevector),
    ] {
        let result = simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .noise(&noise)
            .shots(shots)
            .expect("noisy shots");
        let (mut taken, mut taken_flipped, mut untaken_flipped, mut spectator) = (0, 0, 0, 0);
        for shot in &result.shots {
            if shot[0] {
                taken += 1;
                taken_flipped += usize::from(shot[1]);
            } else {
                untaken_flipped += usize::from(shot[1]);
            }
            spectator += usize::from(shot[2]);
        }
        assert_eq!(untaken_flipped, 0, "{label}: an untaken body fired");
        let rate = taken_flipped as f64 / taken as f64;
        let sigma = (odd * (1.0 - odd) / taken as f64).sqrt();
        assert!(
            (rate - odd).abs() < 5.0 * sigma,
            "{label}: taken flip rate {rate}, exact {odd}"
        );
        let spectator = spectator as f64 / shots as f64;
        assert!(
            (spectator - 0.25).abs() < 5.0 * (0.25f64 * 0.75 / shots as f64).sqrt(),
            "{label}: the event after the region fired on {spectator} of shots"
        );
    }
}

/// Controlled depolarizing on `[control, target]`: the identity when the
/// control reads 0, depolarizing of total rate `p` on the target when it reads 1.
fn controlled_depolarizing(control: usize, target: usize, p: f64) -> NoiseEvent {
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let paulis = [
        [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]],
        [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]],
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]],
    ];
    let mut identity = [[c(0.0, 0.0); 4]; 4];
    let keep = (1.0 - p).sqrt();
    for (t, scale) in [(0, 1.0), (1, 1.0), (2, keep), (3, keep)] {
        identity[t][t] = c(scale, 0.0);
    }
    let mut kraus = vec![identity];
    for pauli in paulis {
        let mut k = [[c(0.0, 0.0); 4]; 4];
        for t in 0..2 {
            for u in 0..2 {
                k[2 + t][2 + u] = pauli[t][u] * (p / 3.0).sqrt();
            }
        }
        kraus.push(k);
    }
    NoiseEvent {
        channel: NoiseChannel::Kraus2q { kraus },
        qubits: smallvec![control, target],
    }
}

// The circuit form of one feed-forward round of the distance-3 repetition code:
// data 0..3 in a GHZ state, two parity checks onto ancillas 3 and 4, then the
// correction the syndrome selects. Syndrome `01` flips data 0, `11` data 1 (as a
// nested region undoing the first flip), and `10` data 2 through the
// one-gate conditional `guarded` lowers it to.
fn feedforward_round() -> Circuit {
    let mut circuit = Circuit::new(5, 5);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Cx, &[0, 2]);
    for (data, ancilla) in [(0, 3), (1, 3), (1, 4), (2, 4)] {
        circuit.add_gate(Gate::Cx, &[data, ancilla]);
    }
    circuit.add_measure(3, 0);
    circuit.add_measure(4, 1);
    circuit.instructions.push(region(
        ClassicalCondition::BitIsOne(0),
        vec![
            gate(Gate::X, &[0]),
            region(
                ClassicalCondition::BitIsOne(1),
                vec![gate(Gate::X, &[0]), gate(Gate::X, &[1])],
            ),
        ],
    ));
    circuit.instructions.extend(guarded(
        ClassicalCondition::RegisterEquals {
            offset: 0,
            size: 2,
            value: 0b10,
        },
        vec![gate(Gate::X, &[2])],
    ));
    for data in 0..3 {
        circuit.add_measure(data, 2 + data);
    }
    circuit
}

// The same round with every guard replaced by control on the unmeasured
// ancillas, and every guarded gate's depolarizing replaced by the same channel
// controlled on the guard (or, with `controlled` false, left unconditional, which
// is what firing an untaken body's noise would mean). Qubits 5 and 6 hold the two compound guards, written
// by noiseless Toffolis. Every measurement is terminal, so the density matrix
// evaluates it exactly.
fn deferred_round(p: f64, controlled: bool) -> (Circuit, NoiseModel) {
    let guard_noise = |control: usize, target: usize| {
        if controlled {
            controlled_depolarizing(control, target, p)
        } else {
            depolarizing(target, p)
        }
    };
    let mut circuit = Circuit::new(7, 0);
    let mut after_gate: Vec<Vec<NoiseEvent>> = Vec::new();
    let toffoli = || Gate::mcu(Gate::X.matrix_2x2(), 2);
    let mut push = |circuit: &mut Circuit, g: Gate, targets: &[usize], events: Vec<NoiseEvent>| {
        circuit.add_gate(g, targets);
        after_gate.push(events);
    };
    let noisy = |targets: &[usize]| targets.iter().map(|&q| depolarizing(q, p)).collect();

    push(&mut circuit, Gate::H, &[0], noisy(&[0]));
    push(&mut circuit, Gate::Cx, &[0, 1], noisy(&[0, 1]));
    push(&mut circuit, Gate::Cx, &[0, 2], noisy(&[0, 2]));
    for (data, ancilla) in [(0, 3), (1, 3), (1, 4), (2, 4)] {
        push(
            &mut circuit,
            Gate::Cx,
            &[data, ancilla],
            noisy(&[data, ancilla]),
        );
    }
    push(&mut circuit, Gate::Cx, &[3, 0], vec![guard_noise(3, 0)]);
    push(&mut circuit, toffoli(), &[3, 4, 5], Vec::new());
    push(&mut circuit, Gate::Cx, &[5, 0], vec![guard_noise(5, 0)]);
    push(&mut circuit, Gate::Cx, &[5, 1], vec![guard_noise(5, 1)]);
    push(&mut circuit, Gate::X, &[3], Vec::new());
    push(&mut circuit, toffoli(), &[3, 4, 6], Vec::new());
    push(&mut circuit, Gate::X, &[3], Vec::new());
    push(&mut circuit, Gate::Cx, &[6, 2], vec![guard_noise(6, 2)]);
    let noise = NoiseModel {
        after_gate,
        readout: Vec::new(),
    };
    (circuit, noise)
}

#[test]
fn a_depolarized_feedforward_round_matches_its_deferred_lowering() {
    let p = 0.08;
    let circuit = feedforward_round();
    let noise = NoiseModel::uniform_depolarizing(&circuit, p);

    let (deferred, deferred_noise) = deferred_round(p, true);
    let exact = simulate(&deferred)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .noise(&deferred_noise)
        .probabilities_of(&[3, 4, 0, 1, 2])
        .expect("exact mixture of the deferred lowering");

    let shots = 40_000;
    for (label, kind) in [
        ("auto", BackendKind::Auto),
        ("statevector", BackendKind::Statevector),
    ] {
        let result = simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .noise(&noise)
            .shots(shots)
            .expect("noisy feed-forward shots");
        assert_within_5_sigma(&histogram(&result.shots, 5), &exact, shots, label);
    }
    let replay = run_shots_noisy(&circuit, &noise, shots, SEED).expect("stabilizer replay");
    let replayed = histogram(&replay.shots, 5);
    assert_within_5_sigma(&replayed, &exact, shots, "replay");

    let (unconditional, unconditional_noise) = deferred_round(p, false);
    let wrong = simulate(&unconditional)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .noise(&unconditional_noise)
        .probabilities_of(&[3, 4, 0, 1, 2])
        .expect("exact mixture with the guarded noise unconditional");
    let separation = replayed
        .iter()
        .zip(&wrong)
        .map(|(&f, &q)| (f - q).abs() / (q * (1.0 - q) / shots as f64).sqrt().max(1e-9))
        .fold(0.0, f64::max);
    assert!(
        separation > 10.0,
        "the comparison cannot tell taken-only noise from unconditional noise: {separation}"
    );
}

#[test]
fn routes_without_a_per_shot_state_decline_a_noisy_guard_by_name() {
    let circuit = feedforward_round();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let err = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .noise(&noise)
        .shots(16)
        .expect_err("the mixture holds every branch at once");
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "density_matrix"),
        "{err:?}"
    );

    let Err(err) = compile_noisy(&circuit, &noise, SEED) else {
        panic!("the compiled sampler fixes its map before a shot");
    };
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "compiled noisy sampler"),
        "{err:?}"
    );

    let mut conditional = Circuit::new(2, 1);
    conditional.add_measure(0, 0);
    conditional.instructions.extend(guarded(
        ClassicalCondition::BitIsOne(0),
        vec![gate(Gate::X, &[1])],
    ));
    conditional.add_measure(1, 0);
    let noise = NoiseModel::uniform_depolarizing(&conditional, 0.01);
    let Err(err) = prism_q::ErrorChainComplex::build(&conditional, &noise, SEED) else {
        panic!("a linear propagation has no branch to skip");
    };
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "error chain complex"),
        "{err:?}"
    );
}

#[test]
fn a_top_level_model_with_events_does_not_validate_on_a_region() {
    let circuit = feedforward_round();
    let mut noise = NoiseModel {
        after_gate: vec![Vec::new(); circuit.instructions.len()],
        readout: Vec::new(),
    };
    noise
        .validate_for(&circuit)
        .expect("readout-only noise keeps the top-level length");
    noise.after_gate[0].push(depolarizing(0, 0.01));
    let err = noise
        .validate_for(&circuit)
        .expect_err("the region bodies would run noiselessly");
    assert!(
        matches!(&err, PrismError::InvalidParameter { message } if message.contains("guarded body slots")),
        "{err:?}"
    );
}

#[test]
fn schedule_rules_decline_a_region() {
    let circuit = feedforward_round();
    let err = NoiseBuilder::new()
        .after_gates(GateFilter::all(), NoiseChannel::Depolarizing { p: 0.01 })
        .on_idle_qubits(NoiseChannel::PhaseDamping { gamma: 0.01 })
        .build(&circuit)
        .expect_err("a region has no static duration");
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "NoiseBuilder"),
        "{err:?}"
    );

    let noise = NoiseBuilder::new()
        .after_gates(GateFilter::all(), NoiseChannel::Depolarizing { p: 0.01 })
        .build(&circuit)
        .expect("gate rules reach guarded gates");
    let uniform = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    assert_eq!(noise.after_gate.len(), uniform.after_gate.len());
    for (built, expected) in noise.after_gate.iter().zip(&uniform.after_gate) {
        assert_eq!(built.len(), expected.len());
    }
}
