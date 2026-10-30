//! Noise on dynamic programs: agreement with the exact mixture of a
//! deferred-measurement lowering, a noisy loop against its analytic law, and the
//! noise a program declines.

use std::f64::consts::PI;

use num_complex::Complex64;
use prism_q::circuit::dynamic::{
    ClassicalExpr, DynamicProgram, DynamicProgramBuilder, RotationKind,
};
use prism_q::{
    BackendKind, Circuit, Gate, GateFilter, NoiseBuilder, NoiseChannel, NoiseEvent, PrismError,
    simulate, simulate_program,
};
use smallvec::smallvec;

const SEED: u64 = 42;

/// Upper 0.1% quantiles of the chi-square distribution, indexed by degrees of
/// freedom. A correct sampler exceeds one at fixed seed with probability 1e-3.
const CHI_SQUARE_999: [f64; 8] = [0.0, 10.828, 13.816, 16.266, 18.467, 20.515, 22.458, 24.322];

/// Histogram of each shot's classical record, bit `i` weighted `1 << i`.
fn histogram(shots: &[Vec<bool>], bits: usize) -> Vec<usize> {
    let mut counts = vec![0; 1 << bits];
    for shot in shots {
        let index = shot[..bits]
            .iter()
            .enumerate()
            .fold(0, |acc, (bit, &set)| acc | (usize::from(set) << bit));
        counts[index] += 1;
    }
    counts
}

/// Pearson's statistic of `counts` against `probabilities`. Every cell must
/// expect at least five shots for the chi-square law to hold.
fn chi_square(counts: &[usize], probabilities: &[f64]) -> f64 {
    let shots: usize = counts.iter().sum();
    counts
        .iter()
        .zip(probabilities)
        .map(|(&seen, &p)| {
            let expected = p * shots as f64;
            assert!(expected >= 5.0, "a cell expects {expected} shots");
            (seen as f64 - expected).powi(2) / expected
        })
        .sum()
}

fn assert_fits(counts: &[usize], probabilities: &[f64], label: &str) {
    let statistic = chi_square(counts, probabilities);
    let critical = CHI_SQUARE_999[probabilities.len() - 1];
    assert!(
        statistic <= critical,
        "{label}: chi-square {statistic:.2} past {critical} over counts {counts:?} \
         against {probabilities:?}"
    );
}

fn teleport_rules(arm_p: f64) -> NoiseBuilder {
    NoiseBuilder::new()
        .after_gates(
            GateFilter::all().named("ry"),
            NoiseChannel::AmplitudeDamping { gamma: 0.3 },
        )
        .after_gates(
            GateFilter::all().named("h"),
            NoiseChannel::PhaseDamping { gamma: 0.2 },
        )
        .after_gates_joint(
            GateFilter::all().named("cx").on_targets([0, 1]),
            NoiseChannel::TwoQubitDepolarizing { p: 0.12 },
        )
        .after_gates(
            GateFilter::all().named("x"),
            NoiseChannel::Depolarizing { p: arm_p },
        )
        .after_gates(
            GateFilter::all().named("z"),
            NoiseChannel::Depolarizing { p: arm_p },
        )
}

const TELEPORT_THETA: f64 = 1.2;

/// Teleport `ry(theta)|0>` from qubit 0 to qubit 2, with each correction a
/// branch of the graph rather than a guarded gate.
fn teleport_program() -> DynamicProgram {
    let mut b = DynamicProgramBuilder::new(3, 3);
    b.add_gate(Gate::Ry(TELEPORT_THETA), &[0])
        .add_gate(Gate::H, &[1])
        .add_gate(Gate::Cx, &[1, 2])
        .add_gate(Gate::Cx, &[0, 1])
        .add_gate(Gate::H, &[0])
        .add_measure(0, 0)
        .add_measure(1, 1);
    b.begin_if(ClassicalExpr::Bit(1));
    b.add_gate(Gate::X, &[2]);
    b.end().unwrap();
    b.begin_if(ClassicalExpr::Bit(0));
    b.add_gate(Gate::Z, &[2]);
    b.end().unwrap();
    b.add_measure(2, 2);
    b.build().unwrap()
}

/// Depolarizing at rate `p` on `target` when `control` is 1, the deferred form
/// of a depolarized gate that runs only on that outcome.
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

/// Exact `(c0, c1, c2)` distribution of the teleport with its corrections
/// deferred to controlled gates. `conditional` keeps the arm noise on the
/// outcome that runs the arm; otherwise it fires on every shot.
fn deferred_teleport(arm_p: f64, conditional: bool) -> Vec<f64> {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::Ry(TELEPORT_THETA), &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::Cz, &[0, 2]);
    let mut noise = teleport_rules(arm_p).build(&circuit).unwrap();
    let arm_noise = |control: usize| {
        if conditional {
            controlled_depolarizing(control, 2, arm_p)
        } else {
            NoiseEvent::pauli(2, arm_p / 3.0, arm_p / 3.0, arm_p / 3.0)
        }
    };
    noise.after_gate[5].push(arm_noise(1));
    noise.after_gate[6].push(arm_noise(0));
    simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .noise(&noise)
        .probabilities_of(&[0, 1, 2])
        .expect("exact mixture of the deferred teleport")
}

#[test]
fn a_noisy_teleport_program_matches_its_deferred_mixture() {
    let arm_p = 0.3;
    let program = teleport_program();
    assert!(program.static_circuit().is_none());
    let rules = teleport_rules(arm_p);
    let exact = deferred_teleport(arm_p, true);

    let shots = 20_000;
    for (label, kind) in [("auto", BackendKind::Auto), ("sparse", BackendKind::Sparse)] {
        let result = simulate_program(&program)
            .backend(kind)
            .noise(&rules)
            .seed(SEED)
            .shots(shots)
            .expect("noisy teleport shots");
        assert_fits(&histogram(&result.shots, 3), &exact, label);
    }

    let unconditional = deferred_teleport(arm_p, false);
    let result = simulate_program(&program)
        .noise(&rules)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    let statistic = chi_square(&histogram(&result.shots, 3), &unconditional);
    assert!(
        statistic > 10.0 * CHI_SQUARE_999[7],
        "the comparison cannot tell arm noise on its own branch from noise on every \
         shot: chi-square {statistic:.2}"
    );
}

/// Prepare `|+>` and measure until the outcome is 0, counting retries mod 4 on
/// qubits 1 (low) and 2 (high).
fn counted_repeat_until_success() -> DynamicProgram {
    let mut b = DynamicProgramBuilder::new(3, 3);
    b.add_gate(Gate::H, &[0]).add_measure(0, 0);
    b.begin_while("retry", ClassicalExpr::Bit(0));
    b.add_gate(Gate::Cx, &[1, 2]).add_gate(Gate::X, &[1]);
    b.add_reset(0).add_gate(Gate::H, &[0]).add_measure(0, 0);
    b.end().unwrap();
    b.add_measure(1, 1).add_measure(2, 2);
    b.build().unwrap()
}

#[test]
fn a_noisy_retry_loop_counts_retries_geometrically() {
    // Amplitude damping after each `h` leaves `P(0) = (1 + gamma) / 2`, so the
    // retry count is geometric with that success rate, and its residue mod 4
    // lands on `r` with probability `p q^r / (1 - q^4)`.
    let gamma = 0.4;
    let rules = NoiseBuilder::new().after_gates(
        GateFilter::all().named("h"),
        NoiseChannel::AmplitudeDamping { gamma },
    );
    let success = (1.0 + gamma) / 2.0;
    let fail: f64 = 1.0 - success;
    let law: Vec<f64> = (0..4)
        .map(|r| success * fail.powi(r) / (1.0 - fail.powi(4)))
        .collect();

    let shots = 10_000;
    let result = simulate_program(&counted_repeat_until_success())
        .noise(&rules)
        .seed(SEED)
        .shots(shots)
        .expect("noisy retry loop");
    assert!(result.shots.iter().all(|shot| !shot[0]));
    let residues: Vec<Vec<bool>> = result.shots.iter().map(|shot| shot[1..].to_vec()).collect();
    assert_fits(&histogram(&residues, 2), &law, "retries mod 4");
}

#[test]
fn a_program_with_one_block_runs_as_its_noisy_circuit() {
    let mut b = DynamicProgramBuilder::new(2, 2);
    b.add_gate(Gate::H, &[0])
        .add_gate(Gate::Cx, &[0, 1])
        .add_measure(0, 0)
        .add_measure(1, 1);
    let program = b.build().unwrap();
    let circuit = program.static_circuit().unwrap();
    let rules = NoiseBuilder::new()
        .after_gates(
            GateFilter::all(),
            NoiseChannel::AmplitudeDamping { gamma: 0.1 },
        )
        .uniform_readout_error(0.02, 0.05);
    let noise = rules.build(circuit).unwrap();
    let dynamic = simulate_program(&program)
        .noise(&rules)
        .seed(SEED)
        .shots(500)
        .unwrap();
    let direct = simulate(circuit)
        .noise(&noise)
        .seed(SEED)
        .shots(500)
        .unwrap();
    assert_eq!(dynamic.shots, direct.shots);
}

#[test]
fn a_runtime_rotation_takes_the_gate_rules() {
    let mut b = DynamicProgramBuilder::new(1, 1);
    b.add_rotation(RotationKind::Rx, &[0], ClassicalExpr::from(PI));
    b.add_measure(0, 0);
    let program = b.build().unwrap();
    assert!(program.static_circuit().is_none());
    let rules = NoiseBuilder::new().after_gates(
        GateFilter::all().named("rx"),
        NoiseChannel::AmplitudeDamping { gamma: 1.0 },
    );
    let shots = simulate_program(&program)
        .noise(&rules)
        .seed(SEED)
        .shots(50)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| !shot[0]));
}

#[test]
fn readout_error_flips_the_record_and_not_the_bit_a_branch_reads() {
    let mut b = DynamicProgramBuilder::new(1, 1);
    b.add_gate(Gate::H, &[0]).add_measure(0, 0);
    b.begin_while("retry", ClassicalExpr::Bit(0));
    b.add_reset(0).add_gate(Gate::H, &[0]).add_measure(0, 0);
    b.end().unwrap();
    let program = b.build().unwrap();
    let rules = NoiseBuilder::new().readout_error(0, 1.0, 0.0);
    let shots = simulate_program(&program)
        .noise(&rules)
        .seed(SEED)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot[0]));
}

fn assert_incompatible(result: prism_q::Result<prism_q::ShotsResult>, needle: &str) {
    match result {
        Err(PrismError::IncompatibleBackend { reason, .. }) => assert!(
            reason.contains(needle),
            "expected a reason naming `{needle}`, got: {reason}"
        ),
        other => panic!("expected IncompatibleBackend, got {other:?}"),
    }
}

#[test]
fn noise_a_program_cannot_carry_is_declined() {
    let program = counted_repeat_until_success();
    let damping = NoiseBuilder::new().after_gates(
        GateFilter::all(),
        NoiseChannel::AmplitudeDamping { gamma: 0.1 },
    );
    assert_incompatible(
        simulate_program(&program)
            .backend(BackendKind::DensityMatrix)
            .noise(&damping)
            .seed(SEED)
            .shots(10),
        "trajector",
    );
    assert_incompatible(
        simulate_program(&program)
            .backend(BackendKind::Stabilizer)
            .noise(&damping)
            .seed(SEED)
            .shots(10),
        "Pauli",
    );

    let leakage =
        NoiseBuilder::new().after_gates(GateFilter::all(), NoiseChannel::Leakage { p: 0.01 });
    assert_incompatible(
        simulate_program(&program)
            .noise(&leakage)
            .seed(SEED)
            .shots(10),
        "leakage",
    );

    let mut b = DynamicProgramBuilder::new(1, 1);
    b.add_rotation(RotationKind::Ry, &[0], ClassicalExpr::from(0.5));
    b.add_measure(0, 0);
    let rotating = b.build().unwrap();
    let over_rotation = NoiseBuilder::new().over_rotation(GateFilter::all(), 0.05);
    assert_incompatible(
        simulate_program(&rotating)
            .noise(&over_rotation)
            .seed(SEED)
            .shots(10),
        "angle",
    );
}

#[test]
fn pauli_noise_runs_a_clifford_program_on_the_tableau() {
    let program = counted_repeat_until_success();
    let rules = NoiseBuilder::new().after_gates(
        GateFilter::all().named("h"),
        NoiseChannel::Depolarizing { p: 0.1 },
    );
    let result = simulate_program(&program)
        .noise(&rules)
        .seed(SEED)
        .shots(200)
        .unwrap();
    assert_eq!(
        result.metadata.backend,
        prism_q::ResolvedBackend::Stabilizer
    );
    assert!(result.shots.iter().all(|shot| !shot[0]));
}
