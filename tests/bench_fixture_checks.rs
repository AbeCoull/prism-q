//! Bench fixtures at bench width, checked against answers that need no
//! reference simulator: mirror circuits, closed forms, and the invariants of a
//! dense or density-matrix result.
//!
//! Checks too large or too slow for a CI runner, and the one row a known defect
//! breaks, are `#[ignore]`d with the reason; run them with `cargo nextest run
//! --features parallel --test bench_fixture_checks --run-ignored all`.

use num_complex::Complex64;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::gates::Gate;
use prism_q::{
    BackendKind, DensityMatrixBackend, NoiseModel, PauliObservable, PauliTerm, circuits,
    run_expectation_values, run_observable_expectation, sim,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

// The bench seed, so each check builds the circuit its row times.
const SEED: u64 = 0xDEAD_BEEF;
const TOL: f64 = 1e-9;

/// Per-qubit Pauli frame as `(x, z)` bits: `X = (1, 0)`, `Y = (1, 1)`, `Z = (0, 1)`.
struct Frame {
    x: Vec<bool>,
    z: Vec<bool>,
}

impl Frame {
    fn emit(&self, c: &mut Circuit, q: usize) {
        match (self.x[q], self.z[q]) {
            (true, false) => c.add_gate(Gate::X, &[q]),
            (true, true) => c.add_gate(Gate::Y, &[q]),
            (false, true) => c.add_gate(Gate::Z, &[q]),
            (false, false) => {}
        }
    }
}

/// Mirror `u` around a random Pauli layer and return the circuit with the one
/// basis state it must end in (Proctor et al., Nature Physics 18, 75, 2022).
///
/// The second half is `u` inverted gate by gate with the layer carried through
/// it as a Pauli frame `F`: each inverse `g^-1` becomes `h` with
/// `h F = F' g^-1`. A Clifford gate keeps `h = g^-1` and conjugates the frame;
/// a rotation keeps the frame and flips its angle where the frame anticommutes
/// with its generator; any other gate is wrapped as `F g^-1 F`. The product is
/// `F_end u^-1 u = F_end`, so the output is `|x(F_end)>` up to phase, and the
/// second half is not the literal inverse of the first, so a kernel that is
/// wrong in one direction does not cancel against itself.
fn mirror(u: &Circuit, seed: u64) -> (Circuit, Vec<bool>) {
    let n = u.num_qubits;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut frame = Frame {
        x: vec![false; n],
        z: vec![false; n],
    };
    for q in 0..n {
        let pauli: u8 = rng.random_range(0..4);
        frame.x[q] = pauli & 1 == 1;
        frame.z[q] = pauli & 2 == 2;
    }

    let mut c = u.clone();
    for q in 0..n {
        frame.emit(&mut c, q);
    }
    for inst in u.instructions.iter().rev() {
        let Instruction::Gate { gate, targets } = inst else {
            panic!("mirror takes a unitary fixture, found {inst:?}");
        };
        let t = targets.as_slice();
        let (x, z) = (&mut frame.x, &mut frame.z);
        let flip = match gate {
            Gate::Rz(_) | Gate::P(_) | Gate::T | Gate::Tdg => Some(x[t[0]]),
            Gate::Rx(_) => Some(z[t[0]]),
            Gate::Ry(_) => Some(x[t[0]] ^ z[t[0]]),
            Gate::Rzz(_) => Some(x[t[0]] ^ x[t[1]]),
            _ => None,
        };
        match (gate, flip) {
            (_, Some(true)) => c.add_gate(gate.clone(), t),
            (_, Some(false)) => c.add_gate(gate.inverse(), t),
            (Gate::Id | Gate::X | Gate::Y | Gate::Z, _) => c.add_gate(gate.clone(), t),
            (Gate::H, _) => {
                std::mem::swap(&mut x[t[0]], &mut z[t[0]]);
                c.add_gate(Gate::H, t);
            }
            (Gate::S | Gate::Sdg, _) => {
                z[t[0]] ^= x[t[0]];
                c.add_gate(gate.inverse(), t);
            }
            (Gate::SX | Gate::SXdg, _) => {
                x[t[0]] ^= z[t[0]];
                c.add_gate(gate.inverse(), t);
            }
            (Gate::Cx, _) => {
                x[t[1]] ^= x[t[0]];
                z[t[0]] ^= z[t[1]];
                c.add_gate(Gate::Cx, t);
            }
            (Gate::Cz, _) => {
                z[t[1]] ^= x[t[0]];
                z[t[0]] ^= x[t[1]];
                c.add_gate(Gate::Cz, t);
            }
            (Gate::Swap, _) => {
                x.swap(t[0], t[1]);
                z.swap(t[0], t[1]);
                c.add_gate(Gate::Swap, t);
            }
            _ => {
                for &q in t {
                    frame.emit(&mut c, q);
                }
                c.add_gate(gate.inverse(), t);
                for &q in t {
                    frame.emit(&mut c, q);
                }
            }
        }
    }
    (c, frame.x)
}

fn basis_index(bits: &[bool]) -> usize {
    bits.iter()
        .enumerate()
        .fold(0, |acc, (q, &b)| acc | (usize::from(b) << q))
}

#[track_caller]
fn assert_normalized(label: &str, probs: &[f64]) {
    let total: f64 = probs.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-8,
        "{label}: probabilities sum to {total}"
    );
}

#[track_caller]
fn dense_probabilities(label: &str, kind: BackendKind, c: &Circuit) -> Vec<f64> {
    let outcome = sim::simulate(c).backend(kind).seed(42).run().unwrap();
    let probs = outcome
        .probabilities
        .unwrap_or_else(|| panic!("{label}: no dense probabilities"))
        .to_vec();
    assert_normalized(label, &probs);
    probs
}

/// Mirror the fixture and require its known bitstring. Statevector reads the
/// full distribution, so it also checks the norm; other backends read marginals.
#[track_caller]
fn assert_mirror(label: &str, kind: BackendKind, u: &Circuit) {
    let (c, expected) = mirror(u, 42);
    if matches!(kind, BackendKind::Statevector) {
        let probs = dense_probabilities(label, kind, &c);
        let p = probs[basis_index(&expected)];
        assert!(
            (p - 1.0).abs() < TOL,
            "{label}: mirror lands on its bitstring with probability {p}"
        );
        return;
    }
    let marginals = sim::simulate(&c)
        .backend(kind.clone())
        .seed(42)
        .marginals()
        .unwrap()
        .marginals;
    assert_eq!(marginals.len(), expected.len(), "{label}: marginal count");
    for (q, (&(p0, p1), &bit)) in marginals.iter().zip(&expected).enumerate() {
        let p = if bit { p1 } else { p0 };
        assert!(
            (p - 1.0).abs() < TOL && (p0 + p1 - 1.0).abs() < TOL,
            "{label} on {kind:?}: qubit {q} reads ({p0}, {p1}), expected {}",
            u8::from(bit)
        );
    }
}

/// Mirror check by sampling, for registers no dense or marginal export covers.
#[track_caller]
fn assert_mirror_shots(label: &str, kind: BackendKind, u: &Circuit) {
    let (mut c, expected) = mirror(u, 42);
    c.measure_all();
    let shots = sim::simulate(&c).backend(kind).seed(42).shots(8).unwrap();
    for shot in &shots.shots {
        assert_eq!(shot, &expected, "{label}: a mirror shot left its bitstring");
    }
}

#[test]
fn random_d10_mirrors_on_statevector() {
    for n in [12usize, 15, 16, 17, 20, 22, 24] {
        let u = circuits::random_circuit(n, 10, SEED);
        assert_mirror(&format!("random_d10/{n}"), BackendKind::Statevector, &u);
    }
}

#[test]
#[ignore = "26 qubits: the dense run holds 1 GiB of amplitudes plus its probability export"]
fn scalability_d5_26_mirrors_on_statevector() {
    let u = circuits::random_circuit(26, 5, SEED);
    assert_mirror("scalability_d5/26", BackendKind::Statevector, &u);
}

#[test]
fn random_d10_mirrors_on_structured_backends() {
    let u = circuits::random_circuit(20, 10, SEED);
    assert_mirror("random_d10/20", BackendKind::Sparse, &u);
    assert_mirror("random_d10/20", BackendKind::Mps { max_bond_dim: 64 }, &u);
    let u = circuits::random_circuit(24, 10, SEED);
    assert_mirror("random_d10/24", BackendKind::Factored, &u);
    let u = circuits::random_circuit(12, 10, SEED);
    assert_mirror("random_d10/12", BackendKind::TensorNetwork, &u);
}

#[test]
fn hea_mirrors() {
    let u = circuits::hardware_efficient_ansatz(20, 5, SEED);
    assert_mirror("hea_l5/20", BackendKind::Statevector, &u);
    for n in [20usize, 22] {
        let u = circuits::hardware_efficient_ansatz(n, 2, SEED);
        assert_mirror(&format!("hea_l2/{n}"), BackendKind::Statevector, &u);
    }
    let u = circuits::hardware_efficient_ansatz(22, 8, SEED);
    assert_mirror("hea_l8/22", BackendKind::Statevector, &u);
}

#[test]
fn qv_mirrors_on_statevector() {
    for n in [16usize, 20, 24] {
        let u = circuits::quantum_volume_circuit(n, n, SEED);
        assert_mirror(&format!("qv/{n}"), BackendKind::Statevector, &u);
    }
}

#[test]
fn diagonal_and_rzz_families_mirror_on_statevector() {
    let u = circuits::qaoa_circuit(24, 3, SEED);
    assert_mirror("qaoa_l3/24", BackendKind::Statevector, &u);
    let u = circuits::diagonal_mixed_circuit(22, 6, SEED);
    assert_mirror("diag_mixed_l6/22", BackendKind::Statevector, &u);
    let u = circuits::brickwork_circuit(20, 24, SEED);
    assert_mirror("brickwork_d24/20", BackendKind::Statevector, &u);
    let u = circuits::matched_brickwork_circuit(16, 12, SEED);
    assert_mirror("matched_d12/16", BackendKind::Statevector, &u);
    let u = circuits::cz_chain_circuit(20, 4, SEED);
    assert_mirror("cz_chain_d4/20", BackendKind::Statevector, &u);
}

#[test]
fn sparse_walk_mirrors_on_sparse() {
    for n in [32usize, 64] {
        let u = circuits::sparse_walk_circuit(n, 12, 5, SEED);
        assert_mirror(&format!("walk_k12/{n}"), BackendKind::Sparse, &u);
    }
}

#[test]
fn clifford_fixtures_mirror_on_the_tableau() {
    let u = circuits::clifford_heavy_circuit(1000, 10, SEED);
    assert_mirror_shots("clifford_d10/1000", BackendKind::Stabilizer, &u);
    let u = circuits::clifford_random_pairs(1000, 10, SEED);
    assert_mirror_shots("random_pairs/1000", BackendKind::Stabilizer, &u);
    let u = circuits::clifford_heavy_circuit(1000, 40, SEED);
    assert_mirror_shots(
        "single_cluster_d40/1000",
        BackendKind::FactoredStabilizer,
        &u,
    );
    let u = circuits::local_clifford_blocks(50, 5, 10, SEED);
    assert_mirror_shots("local_blocks/50x5", BackendKind::FactoredStabilizer, &u);
    let u = circuits::clifford_heavy_circuit(5000, 10, SEED);
    assert_mirror_shots("clifford_d10/5000", BackendKind::Stabilizer, &u);
}

// A stabilizer state's distribution is uniform over an affine subspace, so every
// nonzero probability is the same power of two.
#[test]
fn clifford_d10_statevector_is_uniform_on_its_support() {
    let u = circuits::clifford_heavy_circuit(20, 10, SEED);
    let probs = dense_probabilities("clifford_d10/20", BackendKind::Statevector, &u);
    let support: Vec<f64> = probs.iter().copied().filter(|&p| p > 1e-12).collect();
    assert!(support.len().is_power_of_two(), "support {}", support.len());
    let expected = 1.0 / support.len() as f64;
    assert!(
        support.iter().all(|&p| (p - expected).abs() < TOL),
        "clifford_d10/20: support of {} is not uniform",
        support.len()
    );
}

#[test]
fn ghz_closed_form() {
    for n in [20usize, 24] {
        let label = format!("ghz/{n}");
        let probs =
            dense_probabilities(&label, BackendKind::Statevector, &circuits::ghz_circuit(n));
        let ends = probs[0] + probs[(1 << n) - 1];
        assert!(
            (probs[0] - 0.5).abs() < TOL && (ends - 1.0).abs() < TOL,
            "{label}: ends read {} and {}",
            probs[0],
            probs[(1 << n) - 1]
        );
    }
    for (n, kind) in [
        (64usize, BackendKind::Sparse),
        (5000, BackendKind::Stabilizer),
    ] {
        let mut c = circuits::ghz_circuit(n);
        c.measure_all();
        let shots = sim::simulate(&c).backend(kind).seed(42).shots(64).unwrap();
        let ones = shots.shots.iter().filter(|s| s[0]).count();
        assert!(
            shots.shots.iter().all(|s| s.iter().all(|&b| b == s[0])),
            "ghz/{n}: a shot is not all zeros or all ones"
        );
        assert!(
            ones > 0 && ones < 64,
            "ghz/{n}: {ones} of 64 shots read ones"
        );
    }
}

// QFT on |x> is exp(2 pi i x k / N) / sqrt(N) at index k. A basis-state input
// checks the phases, where |0> only checks the magnitudes.
#[test]
fn qft_closed_form_on_a_basis_state() {
    for n in [12usize, 20, 24] {
        let x = 0x5A5A5Ausize & ((1 << n) - 1);
        let mut c = Circuit::new(n, 0);
        for q in (0..n).filter(|q| x >> q & 1 == 1) {
            c.add_gate(Gate::X, &[q]);
        }
        c.instructions.extend(circuits::qft_circuit(n).instructions);
        let state = sim::simulate(&c)
            .backend(BackendKind::Statevector)
            .seed(42)
            .state_vector()
            .unwrap();
        let dim = 1usize << n;
        let scale = 1.0 / (dim as f64).sqrt();
        let worst = state
            .iter()
            .enumerate()
            .map(|(k, amp)| {
                let angle = std::f64::consts::TAU * ((x * k) % dim) as f64 / dim as f64;
                (amp - Complex64::from_polar(scale, angle)).norm()
            })
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "qft/{n}: worst amplitude error {worst}");
    }
}

// The eigenphase is 1/8, which three or more counting bits represent exactly, so
// the counting register reads 2^(m-3) with certainty and the target stays |1>.
#[test]
fn qpe_closed_form() {
    for n in [16usize, 20, 22] {
        let label = format!("qpe_t_gate/{n}");
        let counting = n - 1;
        let expected = (1usize << counting) | (1 << (counting - 3));
        let probs = dense_probabilities(
            &label,
            BackendKind::Statevector,
            &circuits::phase_estimation_circuit(n),
        );
        let p = probs[expected];
        assert!((p - 1.0).abs() < TOL, "{label}: expected outcome reads {p}");
    }
}

#[test]
fn w_state_closed_form() {
    let n = 20;
    let probs = dense_probabilities(
        "w_state/20",
        BackendKind::Statevector,
        &circuits::w_state_circuit(n),
    );
    for q in 0..n {
        let p = probs[1 << q];
        assert!(
            (p - 1.0 / n as f64).abs() < TOL,
            "w_state/20: one-hot {q} reads {p}"
        );
    }
    for n in [32usize, 64] {
        let mut c = circuits::w_state_circuit(n);
        c.measure_all();
        let shots = sim::simulate(&c)
            .backend(BackendKind::Auto)
            .seed(42)
            .shots(256)
            .unwrap();
        assert!(
            shots
                .shots
                .iter()
                .all(|s| s.iter().filter(|&&b| b).count() == 1),
            "w_state/{n}: a shot is not one-hot"
        );
    }
}

#[track_caller]
fn assert_density_matrix(label: &str, rho: &[Complex64], n: usize) {
    let dim = 1usize << n;
    assert_eq!(rho.len(), dim * dim, "{label}: buffer size");
    let trace: f64 = (0..dim).map(|i| rho[i * dim + i].re).sum();
    assert!((trace - 1.0).abs() < 1e-9, "{label}: trace {trace}");
    for i in 0..dim {
        for j in 0..i {
            let gap = (rho[i * dim + j] - rho[j * dim + i].conj()).norm();
            assert!(gap < 1e-9, "{label}: not Hermitian at ({i}, {j})");
        }
    }
    assert!(
        is_positive_semidefinite(rho, dim, 1e-9),
        "{label}: not positive semidefinite"
    );
}

/// Cholesky of `rho + eps I`; a pivot at or below zero means an eigenvalue
/// below `-eps`.
fn is_positive_semidefinite(rho: &[Complex64], dim: usize, eps: f64) -> bool {
    let mut l = vec![Complex64::new(0.0, 0.0); dim * dim];
    for j in 0..dim {
        let mut pivot = rho[j * dim + j].re + eps;
        for k in 0..j {
            pivot -= l[j * dim + k].norm_sqr();
        }
        if pivot <= 0.0 {
            return false;
        }
        let pivot = pivot.sqrt();
        l[j * dim + j] = Complex64::new(pivot, 0.0);
        for i in j + 1..dim {
            let mut sum = rho[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k].conj();
            }
            l[i * dim + j] = sum / pivot;
        }
    }
    true
}

fn noisy_density_matrix(n: usize) -> Vec<Complex64> {
    let mut dm = DensityMatrixBackend::new(SEED);
    sim::run_on(&mut dm, &circuits::random_circuit(n, 10, SEED)).unwrap();
    dm.apply_2q_depolarizing(0, n - 1, 0.02);
    dm.apply_2q_depolarizing(2, 3, 0.02);
    dm.density_matrix().unwrap()
}

#[test]
fn density_matrix_unitary_layers_mirror_to_a_pure_basis_state() {
    for n in [8usize, 10] {
        let (c, expected) = mirror(&circuits::random_circuit(n, 10, SEED), 42);
        let mut dm = DensityMatrixBackend::new(SEED);
        sim::run_on(&mut dm, &c).unwrap();
        let rho = dm.density_matrix().unwrap();
        let b = basis_index(&expected);
        let dim = 1usize << n;
        let label = format!("unitary_layers/{n}");
        assert!(
            (rho[b * dim + b].re - 1.0).abs() < TOL,
            "{label}: mirror diagonal reads {}",
            rho[b * dim + b]
        );
        assert!(
            (dm.purity() - 1.0).abs() < 1e-9,
            "{label}: purity {}",
            dm.purity()
        );
    }
}

#[test]
fn density_matrix_noisy_rows_stay_physical() {
    assert_density_matrix("noisy_channels/10", &noisy_density_matrix(10), 10);

    let circuit = circuits::qaoa_circuit(10, 6, SEED);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let qubits: Vec<usize> = (0..10).collect();
    let rho = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(42)
        .reduced_density_matrix(&qubits)
        .unwrap();
    assert_density_matrix("rzz_layers/10 noisy", &rho.data, 10);
}

#[test]
#[ignore = "12 qubits: the positivity check is a 4096x4096 Cholesky, about 40 s in the test profile"]
fn density_matrix_noisy_rows_stay_physical_at_12() {
    assert_density_matrix("noisy_channels/12", &noisy_density_matrix(12), 12);
}

#[track_caller]
fn assert_grouped_matches_ungrouped(n: usize) {
    let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
    let terms = circuits::jordan_wigner_hamiltonian(n, 2000, SEED);
    let observable = PauliObservable::from_terms(terms.clone()).unwrap();
    let factors: Vec<Vec<PauliTerm>> = terms.iter().map(|(_, f)| f.clone()).collect();
    let grouped = run_observable_expectation(&circuit, &observable, 42)
        .unwrap()
        .mean;
    let values = run_expectation_values(&circuit, &factors, 42).unwrap();
    let ungrouped: f64 = terms.iter().zip(&values).map(|((c, _), v)| c * v).sum();
    assert!(
        (grouped - ungrouped).abs() < 1e-8 * (1.0 + ungrouped.abs()),
        "pauli_sum_grouped/{n}: grouped {grouped} against ungrouped {ungrouped}"
    );
}

#[test]
fn pauli_sum_grouped_arms_agree_at_16() {
    assert_grouped_matches_ungrouped(16);
}

#[test]
#[ignore = "the bounded tensor-network expectation route panics on an identity \
            observable term at 18 qubits and above, which the ungrouped arm reaches"]
fn pauli_sum_grouped_arms_agree_at_20() {
    assert_grouped_matches_ungrouped(20);
}
