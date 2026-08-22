//! Cross-backend correctness for the density-matrix backend: on unitary
//! circuits it must reproduce the statevector probabilities exactly, and a
//! pure state must stay pure (`Tr(rho^2) == 1`).

mod common;

use common::{SEED, assert_fused_matches_unfused, assert_probs_close};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{circuits, sim};

fn dm_backend(circuit: &Circuit, seed: u64) -> DensityMatrixBackend {
    let mut backend = DensityMatrixBackend::new(seed);
    sim::run_on(&mut backend, circuit).unwrap();
    backend
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn dm_after_channel(
    prep: &[(Gate, usize)],
    kraus: &[[[Complex64; 2]; 2]],
) -> ([[Complex64; 2]; 2], f64) {
    let mut backend = DensityMatrixBackend::new(SEED);
    backend.init(1, 0).unwrap();
    for (gate, q) in prep {
        backend
            .apply(&prism_q::circuit::Instruction::Gate {
                gate: gate.clone(),
                targets: [*q].into_iter().collect(),
            })
            .unwrap();
    }
    backend.apply_1q_kraus(0, kraus);
    let rho = backend.reduced_density_matrix_1q(0).unwrap();
    let trace = rho[0][0].re + rho[1][1].re;
    (rho, trace)
}

fn amplitude_damping(gamma: f64) -> Vec<[[Complex64; 2]; 2]> {
    let s = (1.0 - gamma).sqrt();
    let g = gamma.sqrt();
    vec![
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(s, 0.0)]],
        [[c(0.0, 0.0), c(g, 0.0)], [c(0.0, 0.0), c(0.0, 0.0)]],
    ]
}

fn phase_damping(gamma: f64) -> Vec<[[Complex64; 2]; 2]> {
    let s = (1.0 - gamma).sqrt();
    let g = gamma.sqrt();
    vec![
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(s, 0.0)]],
        [[c(0.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(g, 0.0)]],
    ]
}

const DM_EPS: f64 = 1e-12;

fn statevector_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn run_dm(circuit: &Circuit) -> DensityMatrixBackend {
    let mut backend = DensityMatrixBackend::new(SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend
}

fn dm_probs(circuit: &Circuit) -> Vec<f64> {
    run_dm(circuit).probabilities().unwrap()
}

fn assert_matches_statevector(circuit: &Circuit, label: &str) {
    assert_probs_close(
        &dm_probs(circuit),
        &statevector_probs(circuit),
        DM_EPS,
        label,
    );
}

#[test]
fn dm_all_single_gates_match_statevector() {
    let gates = [
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::SX,
        Gate::SXdg,
        Gate::Rx(0.7),
        Gate::Ry(1.1),
        Gate::Rz(0.3),
        Gate::P(0.5),
    ];
    for gate in gates {
        // The Hadamard sandwich converts phase-only differences into population,
        // so the probability comparison is sensitive to a mishandled conjugation.
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(gate.clone(), &[0]);
        c.add_gate(Gate::H, &[0]);
        assert_matches_statevector(&c, &format!("single_gate {gate:?}"));
    }
}

#[test]
fn dm_two_qubit_gates_match_statevector() {
    for gate in [Gate::Cx, Gate::Cz, Gate::Swap, Gate::Rzz(0.9)] {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Ry(0.6), &[1]);
        c.add_gate(gate.clone(), &[0, 1]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        assert_matches_statevector(&c, &format!("two_qubit_gate {gate:?}"));
    }
}

#[test]
fn dm_bell_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_matches_statevector(&c, "bell");
}

#[test]
fn dm_ghz4_matches_statevector() {
    let c = circuits::ghz_circuit(4);
    assert_matches_statevector(&c, "ghz4");
}

#[test]
fn dm_complex_circuit_matches_statevector() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.5), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(1.3), &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[0, 2]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Rz(0.9), &[1]);
    c.add_gate(Gate::Cx, &[2, 0]);
    assert_matches_statevector(&c, "complex");
}

#[test]
fn dm_qft_matches_statevector() {
    let c = circuits::qft_circuit(5);
    assert_matches_statevector(&c, "qft5");
}

#[test]
fn dm_random_layers_match_statevector() {
    let c = circuits::random_circuit(6, 10, SEED);
    assert_matches_statevector(&c, "random6");
}

#[test]
fn dm_pure_state_stays_pure() {
    let c = circuits::random_circuit(5, 8, SEED);
    let backend = run_dm(&c);
    assert!(
        (backend.purity() - 1.0).abs() < 1e-12,
        "purity of a unitary-evolved state must be 1, got {}",
        backend.purity()
    );
}

#[test]
fn dm_probabilities_sum_to_one() {
    let c = circuits::random_circuit(5, 8, SEED);
    let total: f64 = dm_probs(&c).iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-12,
        "probabilities must sum to 1, got {total}"
    );
}

fn depolarizing(p: f64) -> Vec<[[Complex64; 2]; 2]> {
    let pp = (p / 3.0).sqrt();
    let pi = (1.0 - p).sqrt();
    vec![
        [[c(pi, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(pi, 0.0)]],
        [[c(0.0, 0.0), c(pp, 0.0)], [c(pp, 0.0), c(0.0, 0.0)]],
        [[c(0.0, 0.0), c(0.0, -pp)], [c(0.0, pp), c(0.0, 0.0)]],
        [[c(pp, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-pp, 0.0)]],
    ]
}

#[test]
fn dm_explicit_dispatch_matches_statevector() {
    use prism_q::BackendKind;
    let c = circuits::random_circuit(4, 8, SEED);
    let outcome = sim::simulate(&c)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .run()
        .unwrap();
    let probs = outcome.probabilities.unwrap().to_vec();
    assert_probs_close(&probs, &statevector_probs(&c), DM_EPS, "explicit dispatch");
}

const NOISY_N: usize = 4;
const NOISY_SEEDS: u64 = 64;
const DEPOLARIZING_P: f64 = 0.02;

/// Entangled non-Clifford circuit with terminal measurements, the shape the
/// exact noisy route accepts.
fn noisy_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        c.add_gate(Gate::T, &[q]);
        c.add_gate(Gate::Ry(0.3 + 0.1 * q as f64), &[q]);
    }
    c.measure_all();
    c
}

fn dm_noisy_probs(circuit: &Circuit, noise: &prism_q::NoiseModel, seed: u64) -> Vec<f64> {
    sim::simulate(circuit)
        .backend(prism_q::BackendKind::DensityMatrix)
        .noise(noise)
        .seed(seed)
        .run()
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec()
}

/// Classical-bit pattern as a basis-state index, which is the same number only
/// because `noisy_circuit` measures qubit `q` into bit `q`.
fn outcome_index(shot: &[bool]) -> usize {
    shot.iter()
        .enumerate()
        .filter(|&(_, &b)| b)
        .fold(0usize, |acc, (bit, _)| acc | (1 << bit))
}

fn histogram(shots: &[Vec<bool>], num_outcomes: usize) -> Vec<u64> {
    let mut counts = vec![0u64; num_outcomes];
    for shot in shots {
        counts[outcome_index(shot)] += 1;
    }
    counts
}

fn assert_within_5_sigma(counts: &[u64], exact: &[f64], total: f64, label: &str) {
    for (idx, &count) in counts.iter().enumerate() {
        let p = exact[idx];
        let sigma = (p * (1.0 - p) / total).sqrt();
        let observed = count as f64 / total;
        assert!(
            (observed - p).abs() <= 5.0 * sigma,
            "{label}: outcome {idx} exact {p}, sampled {observed}, tolerance {}",
            5.0 * sigma
        );
    }
}

#[test]
fn dm_exact_noisy_distribution_is_seed_independent_and_matches_the_trajectory_mean() {
    use prism_q::{BackendKind, NoiseModel};
    let circuit = noisy_circuit(NOISY_N);
    let noise = NoiseModel::uniform_depolarizing(&circuit, DEPOLARIZING_P);

    let exact = dm_noisy_probs(&circuit, &noise, SEED);
    for offset in 0..NOISY_SEEDS {
        let repeat = dm_noisy_probs(&circuit, &noise, SEED + offset);
        assert_probs_close(&repeat, &exact, DM_EPS, "exact noisy distribution per seed");
    }

    // Shot `i` of a run seeded `s` draws on `s + i`, so the bases are spaced by
    // the shot count; adjacent ones would replay the same trajectories.
    let shots_per_seed = 250;
    let mut counts = vec![0u64; 1 << NOISY_N];
    for offset in 0..NOISY_SEEDS {
        let result = sim::simulate(&circuit)
            .backend(BackendKind::Statevector)
            .noise(&noise)
            .seed(SEED + offset * shots_per_seed as u64)
            .shots(shots_per_seed)
            .unwrap();
        for (idx, count) in histogram(&result.shots, counts.len()).iter().enumerate() {
            counts[idx] += count;
        }
    }
    let total = (NOISY_SEEDS * shots_per_seed as u64) as f64;
    assert_within_5_sigma(&counts, &exact, total, "trajectory mean vs exact mixture");
}

#[test]
fn dm_noisy_shots_and_counts_sample_the_exact_distribution() {
    use prism_q::{BackendKind, NoiseModel};
    let circuit = noisy_circuit(NOISY_N);
    let noise = NoiseModel::uniform_depolarizing(&circuit, DEPOLARIZING_P);
    let exact = dm_noisy_probs(&circuit, &noise, SEED);

    let num_shots = 20_000;
    let shots = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .shots(num_shots)
        .unwrap();
    assert_eq!(shots.num_shots(), num_shots);
    assert_within_5_sigma(
        &histogram(&shots.shots, exact.len()),
        &exact,
        num_shots as f64,
        "density-matrix shots",
    );

    // Both terminals draw from the one evolution with the same seed, so unlike
    // the noiseless route the histograms must agree exactly, not just in
    // distribution.
    let counts = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .sample_counts(num_shots)
        .unwrap();
    assert_eq!(counts.counts, shots.counts());
    assert_eq!(counts.counts.values().sum::<u64>(), num_shots as u64);
}

#[test]
fn dm_noisy_marginals_agree_with_the_exact_distribution() {
    use prism_q::{BackendKind, NoiseModel};
    let circuit = noisy_circuit(NOISY_N);
    let noise = NoiseModel::uniform_depolarizing(&circuit, DEPOLARIZING_P);
    let exact = dm_noisy_probs(&circuit, &noise, SEED);

    let marginals = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .marginals()
        .unwrap()
        .into_vec();

    for (qubit, &(p0, p1)) in marginals.iter().enumerate() {
        let want: f64 = exact
            .iter()
            .enumerate()
            .filter(|(idx, _)| (idx >> qubit) & 1 == 1)
            .map(|(_, p)| p)
            .sum();
        assert!(
            (p1 - want).abs() < DM_EPS && (p0 + p1 - 1.0).abs() < DM_EPS,
            "qubit {qubit}: marginal ({p0}, {p1}) against exact P(1) = {want}"
        );
    }
}

#[test]
fn dm_expectation_values_agree_with_the_free_function() {
    use prism_q::{BackendKind, NoiseModel, PauliAxis, PauliTerm};
    let mut circuit = Circuit::new(NOISY_N, 0);
    for q in 0..NOISY_N {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 0..NOISY_N - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let observables: Vec<Vec<PauliTerm>> = (0..NOISY_N)
        .flat_map(|q| {
            [PauliAxis::X, PauliAxis::Y, PauliAxis::Z]
                .into_iter()
                .map(move |axis| vec![PauliTerm::new(q, axis)])
        })
        .chain([vec![PauliTerm::z(0), PauliTerm::x(NOISY_N - 1)]])
        .collect();

    let builder = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    let direct =
        prism_q::density_matrix_expectation_values(&circuit, &observables, None, SEED).unwrap();
    assert_probs_close(&builder, &direct, DM_EPS, "noiseless expectation values");

    let noise = NoiseModel::uniform_depolarizing(&circuit, DEPOLARIZING_P);
    let builder_noisy = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    let direct_noisy =
        prism_q::density_matrix_expectation_values(&circuit, &observables, Some(&noise), SEED)
            .unwrap();
    assert_probs_close(
        &builder_noisy,
        &direct_noisy,
        DM_EPS,
        "noisy expectation values",
    );
}

#[test]
fn dm_noisy_terminals_reject_branching_circuits_naming_the_mixture() {
    use prism_q::{BackendKind, NoiseModel};
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_measure(1, 1);
    let noise = NoiseModel::uniform_depolarizing(&circuit, DEPOLARIZING_P);

    let err = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .shots(16)
        .unwrap_err();
    match err {
        prism_q::PrismError::IncompatibleBackend { backend, reason } => {
            assert_eq!(backend, "density_matrix");
            assert!(
                reason.contains("every measurement branch at once"),
                "the rejection must name the mixture, got {reason}"
            );
        }
        other => panic!("expected a named rejection, got {other:?}"),
    }
}

#[test]
fn noisy_exact_terminals_without_a_mixture_name_the_density_matrix() {
    use prism_q::{BackendKind, NoiseModel, PauliTerm};
    let circuit = noisy_circuit(3);
    let noise = NoiseModel::uniform_depolarizing(&circuit, DEPOLARIZING_P);
    let unitary = {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c
    };
    let unitary_noise = NoiseModel::uniform_depolarizing(&unitary, DEPOLARIZING_P);

    let cases: Vec<(&str, prism_q::PrismError)> = vec![
        (
            "run",
            sim::simulate(&circuit)
                .backend(BackendKind::Statevector)
                .noise(&noise)
                .seed(SEED)
                .run()
                .unwrap_err(),
        ),
        (
            "marginals",
            sim::simulate(&circuit)
                .backend(BackendKind::Statevector)
                .noise(&noise)
                .seed(SEED)
                .marginals()
                .unwrap_err(),
        ),
        (
            "expectation_values",
            sim::simulate(&unitary)
                .backend(BackendKind::Statevector)
                .noise(&unitary_noise)
                .seed(SEED)
                .expectation_values(&[vec![PauliTerm::z(0)]])
                .unwrap_err(),
        ),
    ];

    for (terminal, err) in cases {
        match err {
            prism_q::PrismError::IncompatibleBackend { reason, .. } => assert!(
                reason.contains("density-matrix backend"),
                "{terminal}: the rejection must name the exact route, got {reason}"
            ),
            other => panic!("{terminal}: expected a named rejection, got {other:?}"),
        }
    }
}

#[test]
fn noisy_expectation_gradient_is_rejected_naming_the_adjoint() {
    use prism_q::{NoiseModel, PauliTerm};
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::Ry(0.4), &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = NoiseModel::uniform_depolarizing(&circuit, DEPOLARIZING_P);

    let err = sim::simulate(&circuit)
        .noise(&noise)
        .seed(SEED)
        .expectation_gradient(
            &[(1.0, vec![PauliTerm::z(0)])],
            &prism_q::Parameters::new(0),
        )
        .unwrap_err();
    match err {
        prism_q::PrismError::IncompatibleBackend { reason, .. } => assert!(
            reason.contains("adjoint"),
            "the rejection must name the adjoint method, got {reason}"
        ),
        other => panic!("expected a named rejection, got {other:?}"),
    }
}

#[test]
fn dm_noisy_readout_error_flips_sampled_bits() {
    use prism_q::{BackendKind, NoiseModel};
    // The exact distribution describes the state, not the classical outcome, so
    // readout error has to reach the draw rather than the evolution.
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Id, &[0]);
    circuit.add_gate(Gate::Id, &[1]);
    circuit.measure_all();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.with_readout_error(0.3, 0.0);

    let shots = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .shots(20_000)
        .unwrap();
    let ones = shots.shots.iter().filter(|s| s[0]).count() as f64 / 20_000.0;
    assert!(
        (ones - 0.3).abs() < 0.02,
        "readout p01 = 0.3 should flip about 30% of zero outcomes, got {ones}"
    );
}

#[test]
fn dm_amplitude_damping_analytic() {
    let gamma = 0.3;
    // Excited state relaxes: rho11 -> 1 - gamma, rho00 -> gamma.
    let (rho, tr) = dm_after_channel(&[(Gate::X, 0)], &amplitude_damping(gamma));
    assert!((tr - 1.0).abs() < DM_EPS, "trace preserved: {tr}");
    assert!((rho[0][0].re - gamma).abs() < DM_EPS, "ground pop: {rho:?}");
    assert!(
        (rho[1][1].re - (1.0 - gamma)).abs() < DM_EPS,
        "excited pop: {rho:?}"
    );
    // Coherence of |+> decays as sqrt(1-gamma).
    let (rho, tr) = dm_after_channel(&[(Gate::H, 0)], &amplitude_damping(gamma));
    assert!((tr - 1.0).abs() < DM_EPS, "trace preserved: {tr}");
    assert!(
        (rho[0][1].re - 0.5 * (1.0 - gamma).sqrt()).abs() < DM_EPS,
        "coherence: {rho:?}"
    );
}

#[test]
fn dm_phase_damping_analytic() {
    let gamma = 0.4;
    // Populations preserved, off-diagonal decays as sqrt(1-gamma).
    let (rho, tr) = dm_after_channel(&[(Gate::H, 0)], &phase_damping(gamma));
    assert!((tr - 1.0).abs() < DM_EPS, "trace preserved: {tr}");
    assert!(
        (rho[0][0].re - 0.5).abs() < DM_EPS,
        "pop0 preserved: {rho:?}"
    );
    assert!(
        (rho[1][1].re - 0.5).abs() < DM_EPS,
        "pop1 preserved: {rho:?}"
    );
    assert!(
        (rho[0][1].re - 0.5 * (1.0 - gamma).sqrt()).abs() < DM_EPS,
        "coherence: {rho:?}"
    );
    assert!(
        rho[0][1].im.abs() < DM_EPS,
        "no imaginary coherence: {rho:?}"
    );
}

#[test]
fn dm_depolarizing_fixed_point() {
    // Depolarizing with p = 3/4 maps any single-qubit state to I/2.
    for prep in [vec![(Gate::X, 0)], vec![(Gate::H, 0)], vec![(Gate::T, 0)]] {
        let (rho, tr) = dm_after_channel(&prep, &depolarizing(0.75));
        assert!((tr - 1.0).abs() < DM_EPS, "trace preserved: {tr}");
        assert!(
            (rho[0][0].re - 0.5).abs() < DM_EPS,
            "maximally mixed pop0: {rho:?}"
        );
        assert!(
            (rho[1][1].re - 0.5).abs() < DM_EPS,
            "maximally mixed pop1: {rho:?}"
        );
        assert!(rho[0][1].norm() < DM_EPS, "no coherence: {rho:?}");
    }
}

#[test]
fn dm_custom_kraus_matches_amplitude_damping() {
    let gamma = 0.25;
    let (rho, _) = dm_after_channel(&[(Gate::X, 0)], &amplitude_damping(gamma));
    assert!(
        (rho[1][1].re - (1.0 - gamma)).abs() < DM_EPS,
        "custom == AD: {rho:?}"
    );
}

#[test]
fn dm_two_qubit_depolarizing_bell_analytic() {
    // A Bell pair under symmetric two-qubit depolarizing with parameter p:
    // rho' = (1 - p - p/15)|Phi+><Phi+| + (4p/15) I.
    let p = 0.3;
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut backend = DensityMatrixBackend::new(SEED);
    sim::run_on(&mut backend, &c).unwrap();
    backend.apply_2q_depolarizing(0, 1, p);

    let probs = backend.probabilities().unwrap();
    let diag_bell = (1.0 - p - p / 15.0) * 0.5 + 4.0 * p / 15.0;
    let off_bell = 4.0 * p / 15.0;
    assert!((probs[0] - diag_bell).abs() < DM_EPS, "p00: {probs:?}");
    assert!((probs[3] - diag_bell).abs() < DM_EPS, "p11: {probs:?}");
    assert!((probs[1] - off_bell).abs() < DM_EPS, "p01: {probs:?}");
    assert!((probs[2] - off_bell).abs() < DM_EPS, "p10: {probs:?}");
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < DM_EPS, "trace preserved: {total}");
}

fn count_gates(circuit: &Circuit, want: fn(&Gate) -> bool) -> usize {
    circuit
        .instructions
        .iter()
        .filter(
            |inst| matches!(inst, prism_q::circuit::Instruction::Gate { gate, .. } if want(gate)),
        )
        .count()
}

#[test]
fn dm_fused_route_matches_unfused_across_fusion_widths() {
    // 8 is below MIN_QUBITS_FOR_FUSION and fuses nothing, 10 admits one-qubit
    // fusion, 12 the two-qubit and Multi2q passes. MultiFused needs 14 and the
    // diagonal batches 16, both past the 4^n ceiling, so neither is reachable
    // here and neither is covered. The payload assertions are what keep the
    // widths meaningful: a reorder that dropped 12q to Fused2q-only would still
    // match probabilities, and Multi2q ordering is the reason this test exists.
    for (n, want_multi2q, want_fused1q) in [(8usize, 0, 0), (10, 0, 1), (12, 1, 0)] {
        let circuit = circuits::random_circuit(n, 4, SEED);
        let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
        assert!(
            count_gates(&fused, |g| matches!(g, Gate::Multi2q(_))) >= want_multi2q,
            "expected {want_multi2q} Multi2q at {n}q, got {:?}",
            count_gates(&fused, |g| matches!(g, Gate::Multi2q(_)))
        );
        assert!(
            count_gates(&fused, |g| matches!(g, Gate::Fused(_))) >= want_fused1q,
            "expected {want_fused1q} fused 1q run at {n}q"
        );
        if n == 8 {
            assert_eq!(
                fused.instructions.len(),
                circuit.instructions.len(),
                "8q must fuse nothing, it is the control"
            );
        }
        assert_fused_matches_unfused(
            || DensityMatrixBackend::new(SEED),
            &circuit,
            DM_EPS,
            &format!("density matrix fused vs unfused at {n}q"),
        );
    }
}

#[test]
#[should_panic(expected = "distinct targets")]
fn dm_two_qubit_channel_on_a_repeated_qubit_panics() {
    // The block traversal needs four distinct bit positions. NoiseModel rejects
    // this before it reaches a backend, so the panic covers only direct callers
    // of the public kernel.
    let mut backend = DensityMatrixBackend::new(SEED);
    backend.init(4, 0).unwrap();
    backend.apply_2q_depolarizing(2, 2, 0.02);
}

#[test]
#[should_panic(expected = "inside the 4-qubit register")]
fn dm_two_qubit_channel_outside_the_register_panics() {
    let mut backend = DensityMatrixBackend::new(SEED);
    backend.init(4, 0).unwrap();
    backend.apply_2q_depolarizing(0, 4, 0.02);
}

#[test]
fn dm_bare_fused_2q_matches_its_unfused_pair() {
    // A Fused2q that no Multi2q batch absorbs takes conjugate_gate rather than
    // the buffer-conjugation fallback. Built directly, since random_circuit's
    // brick layers batch every 2q run into Multi2q.
    let mat = Gate::Cx.matrix_4x4();
    let mut fused = Circuit::new(3, 0);
    fused.add_gate(Gate::H, &[0]);
    fused.add_gate(Gate::Ry(0.7), &[1]);
    fused.add_gate(Gate::Fused2q(Box::new(mat)), &[0, 1]);
    fused.add_gate(Gate::H, &[2]);

    let mut plain = Circuit::new(3, 0);
    plain.add_gate(Gate::H, &[0]);
    plain.add_gate(Gate::Ry(0.7), &[1]);
    plain.add_gate(Gate::Cx, &[0, 1]);
    plain.add_gate(Gate::H, &[2]);

    assert_probs_close(&dm_probs(&fused), &dm_probs(&plain), DM_EPS, "bare Fused2q");
}

// The 16 Pauli Kraus operators the twirled closed form replaces, weighted
// sqrt(1-p) on I(x)I and sqrt(p/15) elsewhere, indexed 2*bit(q0)+bit(q1).
fn depolarizing_2q_kraus(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let paulis: [[[Complex64; 2]; 2]; 4] = [
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]],
        [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]],
        [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]],
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]],
    ];
    let mut kraus = vec![[[c(0.0, 0.0); 4]; 4]; 16];
    for a in 0..4 {
        for b in 0..4 {
            let w = if a == 0 && b == 0 {
                (1.0 - p).sqrt()
            } else {
                (p / 15.0).sqrt()
            };
            let k = &mut kraus[4 * a + b];
            for (t, row) in k.iter_mut().enumerate() {
                for (tp, entry) in row.iter_mut().enumerate() {
                    *entry = c(w, 0.0) * paulis[a][t >> 1][tp >> 1] * paulis[b][t & 1][tp & 1];
                }
            }
        }
    }
    kraus
}

// Every `n`-qubit Pauli as (xmask, zmask, num_y); the 4^n expectations
// determine rho uniquely.
fn all_pauli_masks(n: usize) -> Vec<(usize, usize, u32)> {
    let d = 1usize << n;
    let mut masks = Vec::with_capacity(d * d);
    for xmask in 0..d {
        for zmask in 0..d {
            masks.push((xmask, zmask, (xmask & zmask).count_ones()));
        }
    }
    masks
}

#[test]
fn dm_two_qubit_depolarizing_matches_kraus_sum() {
    // The closed form and the 16-operator Kraus sum are separate kernels,
    // compared here by full tomography. A pair selects the Kraus sweep's block
    // width, 4 when min(q0,q1) >= 2 and 1 otherwise, and n selects the arm:
    // 2n < 14 is serial. The pairs are each width on each arm. p = 15/16 is the
    // edge where alpha goes to zero.
    for (n, pairs) in [
        (3usize, &[(0usize, 1usize), (1, 2)][..]),
        (5, &[(2, 3)][..]),
        (PAR_N, &[(0, PAR_N - 1), (2, 5)][..]),
    ] {
        let circuit = circuits::random_circuit(n, 4, SEED);
        let masks = all_pauli_masks(n);
        for &(q0, q1) in pairs {
            for p in [0.02, 0.3, 15.0 / 16.0, 1.0] {
                let mut closed = run_dm(&circuit);
                closed.apply_2q_depolarizing(q0, q1, p);
                let mut kraus = run_dm(&circuit);
                kraus.apply_2q_kraus(q0, q1, &depolarizing_2q_kraus(p));

                let got = closed.expectations_pauli(&masks);
                let want = kraus.expectations_pauli(&masks);
                for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (a - b).abs() < DM_EPS,
                        "n={n} pair=({q0},{q1}) p={p} pauli {:?}: closed {a}, kraus {b}",
                        masks[k]
                    );
                }
            }
        }
    }
}

fn diag_2q_kraus(scale: f64, entries: [Complex64; 4]) -> [[Complex64; 4]; 4] {
    let mut m = [[c(0.0, 0.0); 4]; 4];
    for (t, row) in m.iter_mut().enumerate() {
        row[t] = c(scale, 0.0) * entries[t];
    }
    m
}

fn mat4_mul(a: &[[Complex64; 4]; 4], b: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            for k in 0..4 {
                *entry += a[i][k] * b[k][j];
            }
        }
    }
    out
}

// (H (x) I) K (H (x) I) per operator, H on the high `t` bit (q0). H is real and
// self-inverse, so the conjugated channel equals the original sandwiched in H.
fn h_conjugated_on_q0(kraus: &[[[Complex64; 4]; 4]]) -> Vec<[[Complex64; 4]; 4]> {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let mut hi = [[c(0.0, 0.0); 4]; 4];
    for (t, row) in hi.iter_mut().enumerate() {
        for (tp, entry) in row.iter_mut().enumerate() {
            if t & 1 == tp & 1 {
                let sign = if t & 2 != 0 && tp & 2 != 0 { -h } else { h };
                *entry = c(sign, 0.0);
            }
        }
    }
    kraus
        .iter()
        .map(|k| mat4_mul(&mat4_mul(&hi, k), &hi))
        .collect()
}

fn apply_h(backend: &mut DensityMatrixBackend, q: usize) {
    backend
        .apply(&prism_q::circuit::Instruction::Gate {
            gate: Gate::H,
            targets: [q].into_iter().collect(),
        })
        .unwrap();
}

#[test]
fn dm_diagonal_2q_kraus_matches_the_conjugated_dense_route() {
    // An all-diagonal set takes the one-multiply contiguous pass; conjugating
    // every operator by H on q0 fills the superoperator and forces the dense
    // sweep, and sum_k K rho K^dag = H (sum_k K' (H rho H) K'^dag) H relates
    // the two routes exactly. The distinct-phase set has pairwise-distinct
    // off-diagonal slot factors, so a swapped or transposed bit in the
    // diagonal key changes some expectation; the zz pair alone is symmetric
    // under a q0/q1 key swap. The conjugated sets carry order-p off-diagonals,
    // pinning that near-diagonal sets stay on the dense sweep.
    let one = c(1.0, 0.0);
    let phase = |t: f64| c(t.cos(), t.sin());
    for (n, pairs) in [
        (3usize, &[(0usize, 1usize), (1, 2)][..]),
        (5, &[(2, 3)][..]),
        (PAR_N, &[(0, PAR_N - 1), (2, 5)][..]),
    ] {
        let circuit = circuits::random_circuit(n, 4, SEED);
        let masks = all_pauli_masks(n);
        for &(q0, q1) in pairs {
            for p in [0.02f64, 0.3] {
                let sets = [
                    vec![
                        diag_2q_kraus((1.0 - p).sqrt(), [one, one, one, one]),
                        diag_2q_kraus(p.sqrt(), [one, -one, -one, one]),
                    ],
                    vec![
                        diag_2q_kraus((1.0 - p).sqrt(), [one, one, one, one]),
                        diag_2q_kraus(p.sqrt(), [phase(0.0), phase(0.1), phase(0.4), phase(1.0)]),
                    ],
                ];
                for (which, set) in sets.iter().enumerate() {
                    let mut direct = run_dm(&circuit);
                    direct.apply_2q_kraus(q0, q1, set);

                    let mut conjugated = run_dm(&circuit);
                    apply_h(&mut conjugated, q0);
                    conjugated.apply_2q_kraus(q0, q1, &h_conjugated_on_q0(set));
                    apply_h(&mut conjugated, q0);

                    let got = direct.expectations_pauli(&masks);
                    let want = conjugated.expectations_pauli(&masks);
                    for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (a - b).abs() < DM_EPS,
                            "n={n} pair=({q0},{q1}) p={p} set {which} pauli {:?}: \
                             diagonal {a}, dense {b}",
                            masks[k]
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "distinct targets")]
fn dm_diagonal_2q_kraus_on_a_repeated_qubit_panics() {
    // The diagonal dispatch stays behind block_layout's assert, so a repeated
    // target panics rather than scaling the wrong entries.
    let one = c(1.0, 0.0);
    let set = vec![
        diag_2q_kraus((1.0f64 - 0.02).sqrt(), [one, one, one, one]),
        diag_2q_kraus(0.02f64.sqrt(), [one, -one, -one, one]),
    ];
    let mut backend = DensityMatrixBackend::new(SEED);
    backend.init(4, 0).unwrap();
    backend.apply_2q_kraus(2, 2, &set);
}

#[test]
fn dm_measure_basis_state_deterministic() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    let backend = dm_backend(&c, 42);
    assert_eq!(backend.classical_results(), &[true, false]);
    let probs = backend.probabilities().unwrap();
    assert!(
        (probs[1] - 1.0).abs() < DM_EPS,
        "collapsed onto |01>: {probs:?}"
    );
}

#[test]
fn dm_reset_returns_qubit_to_zero() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_reset(0);
    let backend = dm_backend(&c, 42);
    let rdm0 = backend.reduced_density_matrix_1q(0).unwrap();
    assert!(
        (rdm0[0][0].re - 1.0).abs() < DM_EPS,
        "reset qubit population: {rdm0:?}"
    );
    assert!(
        rdm0[1][1].re.abs() < DM_EPS,
        "reset qubit excited population: {rdm0:?}"
    );
    let total: f64 = backend.probabilities().unwrap().iter().sum();
    assert!((total - 1.0).abs() < DM_EPS, "trace preserved: {total}");
}

#[test]
fn dm_reset_from_superposition_gives_maximally_mixed_partial_trace() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_reset(0);
    let backend = dm_backend(&c, 42);
    let rdm1 = backend.reduced_density_matrix_1q(1).unwrap();
    assert!((rdm1[0][0].re - 0.5).abs() < DM_EPS, "rdm1={rdm1:?}");
    assert!((rdm1[1][1].re - 0.5).abs() < DM_EPS, "rdm1={rdm1:?}");
    assert!(
        rdm1[0][1].norm() < DM_EPS,
        "off-diagonal coherence: {rdm1:?}"
    );
}

#[test]
fn dm_conditional_applies_on_measured_one() {
    use prism_q::circuit::{ClassicalCondition, Instruction, SmallVec};
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let targets: SmallVec<[usize; 4]> = [1usize].into_iter().collect();
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets,
    });
    let backend = dm_backend(&c, 42);
    let probs = backend.probabilities().unwrap();
    assert!(
        (probs[3] - 1.0).abs() < DM_EPS,
        "both qubits flipped: {probs:?}"
    );
}

#[test]
fn dm_measurement_marginals_match_statevector() {
    // Measuring a Bell pair yields 00 or 11 with probability 0.5 each; 01 and
    // 10 never occur.
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    let shots = 4000usize;
    let mut counts = [0usize; 4];
    for seed in 0..shots {
        let backend = dm_backend(&c, seed as u64);
        let bits = backend.classical_results();
        let idx = usize::from(bits[0]) | (usize::from(bits[1]) << 1);
        counts[idx] += 1;
    }
    assert_eq!(counts[1], 0, "01 forbidden by correlation");
    assert_eq!(counts[2], 0, "10 forbidden by correlation");
    let p00 = counts[0] as f64 / shots as f64;
    assert!((p00 - 0.5).abs() < 0.05, "p00={p00}");
}

#[test]
fn dm_reduced_density_matrix_matches_statevector() {
    let c = circuits::random_circuit(4, 8, SEED);
    let dm = run_dm(&c);
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &c).unwrap();
    for q in 0..4 {
        let a = dm.reduced_density_matrix_1q(q).unwrap();
        let b = sv.reduced_density_matrix_1q(q).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[i][j] - b[i][j]).norm() < 1e-12,
                    "rdm[{q}][{i}][{j}] mismatch: dm={}, sv={}",
                    a[i][j],
                    b[i][j]
                );
            }
        }
    }
}

// The override against the boxed `Gate::Fused` route it replaces, at both sides
// of the crossover the one-qubit path selects on: below the parallel threshold
// the embedded `2n`-qubit statevector takes two passes, at or above it one block
// superoperator pass. The second matrix is a non-unitary jump branch, so this is
// not restricted to `rho -> U rho U^dagger`.
#[test]
fn dm_apply_1q_matrix_matches_fused_gate_route() {
    let dense = [[c(0.6, 0.0), c(0.8, 0.0)], [c(0.8, 0.0), c(-0.6, 0.0)]];
    let jump = [[c(0.0, 0.0), c(1.3, 0.0)], [c(0.0, 0.0), c(0.0, 0.0)]];

    // 4 qubits embeds an 8-qubit statevector (two-pass path); 7 embeds a
    // 14-qubit one, which is the block-superoperator path.
    for n in [4usize, 7] {
        for (label, matrix) in [("dense", dense), ("jump", jump)] {
            for target in [0usize, n - 1] {
                let circuit = circuits::random_circuit(n, 6, SEED);

                let mut direct = dm_backend(&circuit, SEED);
                direct.apply_1q_matrix(target, &matrix).unwrap();

                let mut boxed = dm_backend(&circuit, SEED);
                boxed
                    .apply(&prism_q::circuit::Instruction::Gate {
                        gate: prism_q::gates::Gate::Fused(Box::new(matrix)),
                        targets: [target].into_iter().collect(),
                    })
                    .unwrap();

                let a = direct.probabilities().unwrap();
                let b = boxed.probabilities().unwrap();
                assert_probs_close(&a, &b, 1e-12, &format!("{label} on q{target} of {n}"));

                for q in 0..n {
                    let ra = direct.reduced_density_matrix_1q(q).unwrap();
                    let rb = boxed.reduced_density_matrix_1q(q).unwrap();
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!(
                                (ra[i][j] - rb[i][j]).norm() < 1e-12,
                                "{label} on q{target} of {n}: rdm[{q}][{i}][{j}] \
                                 direct={} boxed={}",
                                ra[i][j],
                                rb[i][j]
                            );
                        }
                    }
                }
            }
        }
    }
}

// `project`, `apply_reset`, and `apply_2q_depolarizing` walk the `4^n` buffer in
// row-major runs whose stride is the qubit's row bit, and parallelize above the
// embedded statevector's threshold (`2n >= 14`, so n >= 7). At n = 7 the top two
// qubits leave fewer than four row blocks, which is the arm that splits inside a
// block. Each case checks the low qubit and the top qubit.
const PAR_N: usize = 7;

fn pure_reference_state(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.export_statevector().unwrap()
}

/// `run_dm` with a classical bit, which the shared circuit fixtures do not
/// declare because they carry no measurement.
fn run_dm_with_bit(circuit: &Circuit) -> DensityMatrixBackend {
    let mut backend = DensityMatrixBackend::new(SEED);
    backend.init(circuit.num_qubits, 1).unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    backend
}

#[test]
fn dm_measure_collapse_matches_scalar_reference_at_every_target() {
    let circuit = circuits::random_circuit(PAR_N, 4, SEED);
    let state = pure_reference_state(&circuit);

    for target in [0usize, PAR_N - 2, PAR_N - 1] {
        let bit = 1usize << target;
        let mut backend = run_dm_with_bit(&circuit);
        backend
            .apply(&prism_q::circuit::Instruction::Measure {
                qubit: target,
                classical_bit: 0,
            })
            .unwrap();
        let outcome = backend.classical_results()[0];

        let kept: f64 = state
            .iter()
            .enumerate()
            .filter(|(i, _)| (i & bit != 0) == outcome)
            .map(|(_, a)| a.norm_sqr())
            .sum();
        let expected: Vec<f64> = state
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if (i & bit != 0) == outcome {
                    a.norm_sqr() / kept
                } else {
                    0.0
                }
            })
            .collect();
        assert_probs_close(
            &backend.probabilities().unwrap(),
            &expected,
            DM_EPS,
            &format!("collapse on q{target}"),
        );

        // A projected pure state is still pure, which the diagonal alone cannot
        // show: it fails if the off-diagonal runs are scaled or zeroed wrongly.
        assert!(
            (backend.purity() - 1.0).abs() < DM_EPS,
            "collapse on q{target} left purity {}",
            backend.purity()
        );
        let rdm = backend.reduced_density_matrix_1q(target).unwrap();
        let population = if outcome { rdm[1][1].re } else { rdm[0][0].re };
        assert!(
            (population - 1.0).abs() < DM_EPS,
            "collapse on q{target} population {population}"
        );
    }
}

#[test]
fn dm_reset_matches_scalar_reference_at_every_target() {
    let circuit = circuits::random_circuit(PAR_N, 4, SEED);
    let state = pure_reference_state(&circuit);

    for target in [0usize, PAR_N - 2, PAR_N - 1] {
        let bit = 1usize << target;
        let mut backend = run_dm(&circuit);
        backend.reset(target).unwrap();

        // Reset traces the qubit out and reinserts |0>, so the two diagonal
        // entries of each row pair merge into the one with the bit clear.
        let expected: Vec<f64> = state
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if i & bit != 0 {
                    0.0
                } else {
                    a.norm_sqr() + state[i | bit].norm_sqr()
                }
            })
            .collect();
        assert_probs_close(
            &backend.probabilities().unwrap(),
            &expected,
            DM_EPS,
            &format!("reset on q{target}"),
        );

        let rdm = backend.reduced_density_matrix_1q(target).unwrap();
        assert!(
            (rdm[0][0].re - 1.0).abs() < DM_EPS && rdm[0][1].norm() < DM_EPS,
            "reset on q{target} left rdm {rdm:?}"
        );

        // Tracing one qubit out cannot change any other qubit's reduced state.
        let mut sv = StatevectorBackend::new(SEED);
        sim::run_on(&mut sv, &circuit).unwrap();
        for q in (0..PAR_N).filter(|&q| q != target) {
            let a = backend.reduced_density_matrix_1q(q).unwrap();
            let b = sv.reduced_density_matrix_1q(q).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (a[i][j] - b[i][j]).norm() < DM_EPS,
                        "reset on q{target}: rdm[{q}][{i}][{j}] dm={} sv={}",
                        a[i][j],
                        b[i][j]
                    );
                }
            }
        }
    }
}

#[test]
fn dm_two_qubit_depolarizing_matches_analytic_above_parallel_threshold() {
    // The Pauli twirl gives rho -> (1-lambda) rho + lambda (I/4) (x) tr_{q0,q1} rho
    // with lambda = 16p/15, so each reduced state mixes toward I/2 by lambda and
    // qubits outside the pair are untouched.
    let p = 0.3;
    let lambda = 16.0 * p / 15.0;
    let circuit = circuits::random_circuit(PAR_N, 4, SEED);

    for (q0, q1) in [(0usize, 1usize), (0, PAR_N - 1)] {
        let mut sv = StatevectorBackend::new(SEED);
        sim::run_on(&mut sv, &circuit).unwrap();

        let mut backend = run_dm(&circuit);
        backend.apply_2q_depolarizing(q0, q1, p);

        for q in 0..PAR_N {
            let before = sv.reduced_density_matrix_1q(q).unwrap();
            let after = backend.reduced_density_matrix_1q(q).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    let mixed = if i == j { c(0.5, 0.0) } else { c(0.0, 0.0) };
                    let expected = if q == q0 || q == q1 {
                        before[i][j] * (1.0 - lambda) + mixed * lambda
                    } else {
                        before[i][j]
                    };
                    assert!(
                        (after[i][j] - expected).norm() < DM_EPS,
                        "depolarizing({q0},{q1}): rdm[{q}][{i}][{j}] expected {expected}, \
                         got {}",
                        after[i][j]
                    );
                }
            }
        }
    }
}

#[test]
fn dm_purity_matches_analytic_above_parallel_reduce_threshold() {
    // 4^8 entries clears the reduction's parallel threshold. One qubit fully
    // depolarized against seven in |0> gives rho = (I/2) (x) |0><0|^7, purity 1/2.
    let n = 8usize;
    let mut backend = DensityMatrixBackend::new(SEED);
    backend.init(n, 0).unwrap();
    assert!(
        (backend.purity() - 1.0).abs() < DM_EPS,
        "pure |0...0> purity {}",
        backend.purity()
    );

    backend.apply_1q_kraus(0, &depolarizing(0.75));
    assert!(
        (backend.purity() - 0.5).abs() < DM_EPS,
        "one maximally mixed qubit gives purity 1/2, got {}",
        backend.purity()
    );
}

fn assert_rdm_close(a: &DensityMatrixBackend, b: &StatevectorBackend, n: usize, label: &str) {
    for q in 0..n {
        let x = a.reduced_density_matrix_1q(q).unwrap();
        let y = b.reduced_density_matrix_1q(q).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (x[i][j] - y[i][j]).norm() < DM_EPS,
                    "{label}: rdm[{q}][{i}][{j}] dm={} sv={}",
                    x[i][j],
                    y[i][j]
                );
            }
        }
    }
}

// `QftBlock` holds its qubit range in the variant, so the ket-register offset
// the sandwich applies to `targets` has to reach `start` too. The QFT of a basis
// state is uniform over the register under either bit-order convention, which an
// identity-acting block cannot produce.
#[test]
fn dm_qft_block_through_apply_evolves_the_state() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::QftBlock { start: 0, num: 2 }, &[0, 1]);

    let mut dm = DensityMatrixBackend::new(SEED);
    dm.init(2, 0).unwrap();
    dm.apply_instructions(&circuit.instructions).unwrap();

    assert_probs_close(
        &dm.probabilities().unwrap(),
        &[0.25; 4],
        DM_EPS,
        "qft block of a basis state",
    );
    assert!(
        (dm.purity() - 1.0).abs() < DM_EPS,
        "qft block keeps the state pure, purity {}",
        dm.purity()
    );

    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();
    assert_rdm_close(&dm, &sv, 2, "qft block");
}

// `BatchPhase` keeps its control in `targets[0]` and its targets in the payload,
// so only the control picks up the ket-register offset. On the uniform two-qubit
// superposition a cphase(pi/2) from q0 to q1 sends the amplitude 1/2 at index 3
// to i/2. The result is symmetric in the pair, so both reduced states are
// [[1/2, (1-i)/4], [(1+i)/4, 1/2]].
#[test]
fn dm_batch_phase_through_apply_matches_the_unfused_cphase() {
    let phase = c(0.0, 1.0);
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);

    let mut dm = DensityMatrixBackend::new(SEED);
    dm.init(2, 0).unwrap();
    dm.apply_instructions(&circuit.instructions).unwrap();
    dm.apply(&prism_q::circuit::Instruction::Gate {
        gate: Gate::BatchPhase(Box::new(prism_q::gates::BatchPhaseData {
            phases: [(1usize, phase)].into_iter().collect(),
        })),
        targets: [0usize].into_iter().collect(),
    })
    .unwrap();

    let expected = [[c(0.5, 0.0), c(0.25, -0.25)], [c(0.25, 0.25), c(0.5, 0.0)]];
    for q in 0..2 {
        let rdm = dm.reduced_density_matrix_1q(q).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (rdm[i][j] - expected[i][j]).norm() < DM_EPS,
                    "batch phase: rdm[{q}][{i}][{j}] expected {} got {}",
                    expected[i][j],
                    rdm[i][j]
                );
            }
        }
    }

    circuit.add_gate(
        Gate::cu([[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), phase]]),
        &[0, 1],
    );
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();
    assert_rdm_close(&dm, &sv, 2, "batch phase");
}

// A 1q gate and the 1q channel that follows it compose into one superoperator
// sweep on the noisy density-matrix route. The composed map is order sensitive,
// so comparing every 1q Pauli expectation against sequential application pins
// the order as well as the value: the three expectations fix each reduced
// density matrix completely.
#[test]
fn fused_gate_and_channel_matches_sequential_application() {
    use prism_q::sim::noise::{NoiseModel, density_matrix_expectation_values};
    use prism_q::sim::unified_pauli::PauliTerm;

    let gamma = 0.17;
    let n = 4;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 0..n {
        circuit.add_gate(Gate::Ry(0.63 + q as f64), &[q]);
    }
    for q in 0..n {
        circuit.add_gate(Gate::S, &[q]);
    }

    let noise = NoiseModel::with_amplitude_damping(&circuit, gamma);
    let observables: Vec<Vec<PauliTerm>> = (0..n)
        .flat_map(|q| {
            [
                vec![PauliTerm::x(q)],
                vec![PauliTerm::y(q)],
                vec![PauliTerm::z(q)],
            ]
        })
        .collect();
    let fused =
        density_matrix_expectation_values(&circuit, &observables, Some(&noise), SEED).unwrap();

    // Sequential reference: gate, then the channel, as two separate sweeps.
    let damping = [
        [c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c((1.0 - gamma).sqrt(), 0.0)],
    ];
    let jump = [
        [c(0.0, 0.0), c(gamma.sqrt(), 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0)],
    ];
    let mut reference = DensityMatrixBackend::new(SEED);
    reference.init(n, 0).unwrap();
    for instr in &circuit.instructions {
        reference.apply(instr).unwrap();
        if let prism_q::circuit::Instruction::Gate { targets, .. } = instr {
            for &q in targets.iter() {
                reference.apply_1q_kraus(q, &[damping, jump]);
            }
        }
    }

    for q in 0..n {
        let rdm = reference.reduced_density_matrix_1q(q).unwrap();
        let expected = [
            (rdm[0][1] + rdm[1][0]).re,
            (Complex64::new(0.0, 1.0) * (rdm[0][1] - rdm[1][0])).re,
            (rdm[0][0] - rdm[1][1]).re,
        ];
        for (k, label) in ["X", "Y", "Z"].iter().enumerate() {
            let got = fused[3 * q + k];
            assert!(
                (got - expected[k]).abs() < DM_EPS,
                "qubit {q} <{label}>: fused {got} vs sequential {}",
                expected[k]
            );
        }
    }
}
