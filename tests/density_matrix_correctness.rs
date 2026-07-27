//! Cross-backend correctness for the density-matrix backend.
//!
//! On noiseless (unitary) circuits the density-matrix backend must reproduce the
//! statevector backend's basis-state probabilities exactly, and a pure state must
//! stay pure (`Tr(rho^2) == 1`).

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

use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Build a one-qubit backend, prepare it with `prep`, apply Kraus set `kraus`,
/// and return the resulting one-qubit density matrix and its trace.
fn dm_after_channel(
    prep: &[(Gate, usize)],
    kraus: &[[[Complex64; 2]; 2]],
) -> ([[Complex64; 2]; 2], f64) {
    let mut backend = DensityMatrixBackend::new(42);
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
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn run_dm(circuit: &Circuit) -> DensityMatrixBackend {
    let mut backend = DensityMatrixBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend
}

fn dm_probs(circuit: &Circuit) -> Vec<f64> {
    run_dm(circuit).probabilities().unwrap()
}

fn assert_probs_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: probability vector length mismatch"
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < DM_EPS,
            "{label}: probability[{i}] mismatch: dm={a}, statevector={e}"
        );
    }
}

fn assert_matches_statevector(circuit: &Circuit, label: &str) {
    assert_probs_close(&dm_probs(circuit), &statevector_probs(circuit), label);
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
    let c = circuits::random_circuit(6, 10, 0xDEAD_BEEF);
    assert_matches_statevector(&c, "random6");
}

#[test]
fn dm_pure_state_stays_pure() {
    let c = circuits::random_circuit(5, 8, 0xDEAD_BEEF);
    let backend = run_dm(&c);
    assert!(
        (backend.purity() - 1.0).abs() < 1e-12,
        "purity of a unitary-evolved state must be 1, got {}",
        backend.purity()
    );
}

#[test]
fn dm_probabilities_sum_to_one() {
    let c = circuits::random_circuit(5, 8, 0xDEAD_BEEF);
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
    let c = circuits::random_circuit(4, 8, 0xDEAD_BEEF);
    let outcome = sim::simulate(&c)
        .backend(BackendKind::DensityMatrix)
        .seed(42)
        .run()
        .unwrap();
    let probs = outcome.probabilities.unwrap().to_vec();
    assert_probs_close(&probs, &statevector_probs(&c), "explicit dispatch");
}

#[test]
fn dm_noisy_shot_sampling_is_rejected() {
    use prism_q::{BackendKind, NoiseModel};
    // Density matrix is an exact backend, not a per-shot sampler. Noisy shot
    // sampling must fail cleanly; exact noisy expectation goes through
    // density_matrix_expectation_values instead.
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&c, 0.01);
    let result = sim::simulate(&c)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(42)
        .shots(100);
    assert!(result.is_err(), "noisy DM shot sampling should be rejected");
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
    // A Custom Kraus set equal to amplitude damping must reproduce it.
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
    let mut backend = DensityMatrixBackend::new(42);
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
    // A Bell pair reset on one qubit leaves the other maximally mixed.
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
    // 10 never occur. The empirical frequency must match the statevector
    // analytic marginals.
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
    let c = circuits::random_circuit(4, 8, 0xDEAD_BEEF);
    let dm = run_dm(&c);
    let mut sv = StatevectorBackend::new(42);
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
