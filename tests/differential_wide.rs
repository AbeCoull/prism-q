//! Seeded circuits per class at 10, 14 and 18 qubits, the widths where the fusion passes
//! and the tiled statevector kernels switch on, run on every backend that accepts them and
//! compared against the statevector, whose fused run is first checked against its own
//! unfused apply loop.

mod common;

use common::{
    DM_EPS, FACTORED_EPS, MPS_EPS, PRODUCT_EPS, SEED, SPARSE_EPS, STAB_EPS, SV_EPS, TN_EPS,
    assert_probs_close, run_fused_probs, run_unfused_probs,
};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::fusion::fuse_circuit_for_width;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::{BackendKind, SpdTruncation, simulate};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::TAU;

// Stabilizer rank and Pauli propagation sum weighted branches or terms, so neither shares
// a bound with an amplitude backend.
const BRANCH_SUM_EPS: f64 = 1e-10;

const WIDTHS: [usize; 3] = [10, 14, 18];

// The stabilizer-rank expansion costs 2^t branches, so it and the exact Pauli sum run only
// up to this T count.
const MAX_T_COUNT: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    QuantumVolume,
    MatchedBrickwork,
    LayeredMixed,
    Hea,
    Qaoa,
    DiagonalMixed,
    Qft,
    QftTextbook,
    Random,
    Clifford,
    CliffordT,
    Rotations,
}

impl Class {
    const ALL: [Class; 12] = [
        Class::QuantumVolume,
        Class::MatchedBrickwork,
        Class::LayeredMixed,
        Class::Hea,
        Class::Qaoa,
        Class::DiagonalMixed,
        Class::Qft,
        Class::QftTextbook,
        Class::Random,
        Class::Clifford,
        Class::CliffordT,
        Class::Rotations,
    ];

    fn build(self, n: usize) -> Circuit {
        match self {
            Class::QuantumVolume => circuits::quantum_volume_circuit(n, 3, SEED),
            Class::MatchedBrickwork => circuits::matched_brickwork_circuit(n, 4, SEED),
            Class::LayeredMixed => layered_mixed(n, 6),
            Class::Hea => circuits::hardware_efficient_ansatz(n, 3, SEED),
            Class::Qaoa => circuits::qaoa_circuit(n, 2, SEED),
            Class::DiagonalMixed => circuits::diagonal_mixed_circuit(n, 4, SEED),
            Class::Qft => prepared_qft(n),
            Class::QftTextbook => {
                prism_q::circuit::expand_qft_blocks(&prepared_qft(n)).into_owned()
            }
            Class::Random => circuits::random_circuit(n, 6, SEED),
            Class::Clifford => circuits::clifford_random_pairs(n, 6, SEED),
            Class::CliffordT => clifford_with_mid_t(n),
            Class::Rotations => circuits::single_qubit_rotation_circuit(n, 5, SEED),
        }
    }
}

// Seeded 2q gates of mixed kinds on a fresh random matching each layer, then rotations on
// a random subset of the register, so some 2q gates carry a trailing 1q run and others
// sit next to a layer they commute across.
fn layered_mixed(n: usize, layers: usize) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(rng.random::<f64>() * TAU), &[q]);
    }
    for _ in 0..layers {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.random_range(0..=i));
        }
        for pair in order.chunks_exact(2) {
            let theta = rng.random::<f64>() * TAU;
            let gate = match rng.random_range(0..5) {
                0 => Gate::Cx,
                1 => Gate::Cz,
                2 => Gate::Rzz(theta),
                3 => Gate::cphase(theta),
                _ => Gate::Swap,
            };
            c.add_gate(gate, pair);
        }
        for q in 0..n {
            if rng.random_bool(0.5) {
                let theta = rng.random::<f64>() * TAU;
                let gate = match rng.random_range(0..3) {
                    0 => Gate::Rx(theta),
                    1 => Gate::Ry(theta),
                    _ => Gate::Rz(theta),
                };
                c.add_gate(gate, &[q]);
            }
        }
    }
    c
}

// A QFT of the all-zero state is uniform in probability, so a product state goes in first.
fn prepared_qft(n: usize) -> Circuit {
    let mut c = circuits::single_qubit_rotation_circuit(n, 1, SEED);
    c.instructions.extend(circuits::qft_circuit(n).instructions);
    c
}

fn clifford_with_mid_t(n: usize) -> Circuit {
    let body = circuits::clifford_random_pairs(n, 6, SEED);
    let mid = body.instructions.len() / 2;
    let mut c = Circuit::new(n, 0);
    c.instructions.extend_from_slice(&body.instructions[..mid]);
    for q in [0, n / 3, 2 * n / 3, n - 1] {
        c.add_gate(Gate::T, &[q]);
    }
    c.instructions.extend_from_slice(&body.instructions[mid..]);
    c
}

// Stochastic Pauli propagation and the bounded tensor network answer approximately, so
// neither takes part. Every participant except the stabilizer pair, stabilizer rank and
// Pauli propagation runs the same fused stream as the statevector; the unfused
// statevector run in `reference` is what checks the fused stream itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Participant {
    Sparse,
    Mps,
    TensorNetwork,
    Factored,
    Stabilizer,
    FactoredStabilizer,
    Product,
    DensityMatrix,
    StabilizerRank,
    DeterministicPauli,
}

impl Participant {
    const ALL: [Participant; 10] = [
        Participant::Sparse,
        Participant::Mps,
        Participant::TensorNetwork,
        Participant::Factored,
        Participant::Stabilizer,
        Participant::FactoredStabilizer,
        Participant::Product,
        Participant::DensityMatrix,
        Participant::StabilizerRank,
        Participant::DeterministicPauli,
    ];

    // Costs quoted are per case in the test profile on a four-core i7-6700K.
    fn skip(self, class: Class, circuit: &Circuit) -> Option<&'static str> {
        use Class::*;
        let n = circuit.num_qubits;
        let low_t = circuit.is_clifford_plus_t() && circuit.t_count() <= MAX_T_COUNT;
        match self {
            Participant::Sparse
                if n >= 18
                    && !matches!(
                        class,
                        DiagonalMixed | Random | Clifford | CliffordT | Rotations
                    ) =>
            {
                Some(
                    "a dense 2^18 map costs 0.4 to 2.4 s, and on two classes the default prune \
                     threshold drops real amplitude (the ignored sparse_* tests pin both)",
                )
            }
            Participant::Mps
                if n >= 18
                    && matches!(
                        class,
                        QuantumVolume
                            | MatchedBrickwork
                            | LayeredMixed
                            | DiagonalMixed
                            | Qft
                            | QftTextbook
                    ) =>
            {
                Some("long-range pairs push the exact bond toward 2^9, 0.7 to 14 s")
            }
            Participant::DensityMatrix if n > 10 => Some("4^n amplitudes, 4 GiB at 14 qubits"),
            Participant::DensityMatrix if !matches!(class, QuantumVolume | Qaoa | Rotations) => {
                Some(
                    "0.5 to 2.4 s at 10 qubits; the kept classes reach Multi2q, BatchRzz and \
                     MultiFused at the 20-qubit fusion width",
                )
            }
            Participant::Stabilizer | Participant::FactoredStabilizer => {
                (!circuit.is_clifford_only()).then_some("non-Clifford gates")
            }
            Participant::Product => circuit.has_entangling_gates().then_some("entangling gates"),
            Participant::StabilizerRank if circuit.t_count() == 0 => Some("no T gates"),
            Participant::StabilizerRank | Participant::DeterministicPauli => {
                (!low_t).then_some("off the Clifford+T grid or above the T budget")
            }
            _ => None,
        }
    }

    fn check(self, circuit: &Circuit, reference: &Reference, label: &str) {
        let label = format!("{self:?} {label}");
        match self {
            Participant::Sparse => {
                let backend = SparseBackend::new(SEED);
                compare_state(backend, circuit, reference, SPARSE_EPS, &label)
            }
            Participant::Mps => {
                let exact_bond = 1usize << (circuit.num_qubits / 2);
                let backend = MpsBackend::new(SEED, exact_bond);
                compare_state(backend, circuit, reference, MPS_EPS, &label)
            }
            Participant::TensorNetwork => {
                let backend = TensorNetworkBackend::new(SEED);
                compare_state(backend, circuit, reference, TN_EPS, &label)
            }
            Participant::Factored => {
                let backend = FactoredBackend::new(SEED);
                compare_state(backend, circuit, reference, FACTORED_EPS, &label)
            }
            Participant::Stabilizer => {
                let backend = StabilizerBackend::new(SEED);
                compare_state(backend, circuit, reference, STAB_EPS, &label)
            }
            Participant::FactoredStabilizer => {
                let backend = FactoredStabilizerBackend::new(SEED);
                compare_state(backend, circuit, reference, STAB_EPS, &label)
            }
            Participant::Product => {
                let backend = ProductStateBackend::new(SEED);
                compare_state(backend, circuit, reference, PRODUCT_EPS, &label)
            }
            Participant::DensityMatrix => {
                let probs = run_fused_probs(&mut DensityMatrixBackend::new(SEED), circuit);
                assert_probs_close(&probs, &reference.probs, DM_EPS, &label);
            }
            Participant::StabilizerRank => {
                let outcome = simulate(circuit)
                    .backend(BackendKind::StabilizerRank)
                    .seed(SEED)
                    .run()
                    .unwrap();
                let probs = outcome.probabilities.expect("probabilities").to_vec();
                assert_probs_close(&probs, &reference.probs, BRANCH_SUM_EPS, &label);
            }
            Participant::DeterministicPauli => {
                let marginals = simulate(circuit)
                    .backend(BackendKind::DeterministicPauli {
                        truncation: SpdTruncation::Threshold {
                            epsilon: 0.0,
                            max_terms: 0,
                        },
                    })
                    .seed(SEED)
                    .marginals()
                    .unwrap()
                    .into_vec();
                let p1: Vec<f64> = marginals.iter().map(|&(_, p1)| p1).collect();
                assert_probs_close(&p1, &reference.marginals(), BRANCH_SUM_EPS, &label);
            }
        }
    }
}

struct Reference {
    probs: Vec<f64>,
    amplitudes: Vec<Complex64>,
}

impl Reference {
    fn new(circuit: &Circuit, label: &str) -> Self {
        let mut fused = StatevectorBackend::new(SEED);
        let probs = run_fused_probs(&mut fused, circuit);
        let amplitudes = fused.export_statevector().unwrap();
        let mut unfused = StatevectorBackend::new(SEED);
        run_unfused_probs(&mut unfused, circuit);
        assert_amplitudes_close(
            &amplitudes,
            &unfused.export_statevector().unwrap(),
            SV_EPS,
            &format!("fused statevector {label}"),
        );
        Self { probs, amplitudes }
    }

    fn marginals(&self) -> Vec<f64> {
        let n = self.probs.len().trailing_zeros() as usize;
        (0..n)
            .map(|q| {
                self.probs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i >> q) & 1 == 1)
                    .map(|(_, p)| p)
                    .sum()
            })
            .collect()
    }
}

fn compare_state<B: Backend>(
    mut backend: B,
    circuit: &Circuit,
    reference: &Reference,
    eps: f64,
    label: &str,
) {
    let probs = run_fused_probs(&mut backend, circuit);
    assert_probs_close(&probs, &reference.probs, eps, label);
    let amplitudes = backend.export_statevector().unwrap();
    assert_amplitudes_close(&amplitudes, &reference.amplitudes, eps, label);
}

// Aligns `actual` to `expected` by the phase of their overlap, so a backend that tracks
// the global phase differently still compares amplitude by amplitude.
fn assert_amplitudes_close(actual: &[Complex64], expected: &[Complex64], eps: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: amplitude count");
    let overlap: Complex64 = actual.iter().zip(expected).map(|(a, e)| a.conj() * e).sum();
    let phase = overlap / overlap.norm();
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a * phase - e).norm();
        assert!(
            diff < eps,
            "{label} amp[{i}]: expected {e:.12}, got {:.12} (diff {diff:.2e}, eps {eps:.0e})",
            a * phase
        );
    }
}

fn check_case(class: Class, n: usize) {
    let circuit = class.build(n);
    let label = format!("{class:?} {n}q");
    let reference = Reference::new(&circuit, &label);
    for participant in Participant::ALL {
        if participant.skip(class, &circuit).is_none() {
            participant.check(&circuit, &reference, &label);
        }
    }
}

fn check_one(participant: Participant, class: Class, n: usize) {
    let circuit = class.build(n);
    let label = format!("{class:?} {n}q");
    participant.check(&circuit, &Reference::new(&circuit, &label), &label);
}

macro_rules! differential_cases {
    ($($name:ident => ($class:expr, $n:expr)),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                check_case($class, $n);
            }
        )+
    };
}

differential_cases! {
    quantum_volume_10q => (Class::QuantumVolume, 10),
    quantum_volume_14q => (Class::QuantumVolume, 14),
    quantum_volume_18q => (Class::QuantumVolume, 18),
    matched_brickwork_10q => (Class::MatchedBrickwork, 10),
    matched_brickwork_14q => (Class::MatchedBrickwork, 14),
    matched_brickwork_18q => (Class::MatchedBrickwork, 18),
    layered_mixed_10q => (Class::LayeredMixed, 10),
    layered_mixed_14q => (Class::LayeredMixed, 14),
    layered_mixed_18q => (Class::LayeredMixed, 18),
    hea_10q => (Class::Hea, 10),
    hea_14q => (Class::Hea, 14),
    hea_18q => (Class::Hea, 18),
    qaoa_10q => (Class::Qaoa, 10),
    qaoa_14q => (Class::Qaoa, 14),
    qaoa_18q => (Class::Qaoa, 18),
    diagonal_mixed_10q => (Class::DiagonalMixed, 10),
    diagonal_mixed_14q => (Class::DiagonalMixed, 14),
    diagonal_mixed_18q => (Class::DiagonalMixed, 18),
    qft_10q => (Class::Qft, 10),
    qft_14q => (Class::Qft, 14),
    qft_18q => (Class::Qft, 18),
    qft_textbook_10q => (Class::QftTextbook, 10),
    qft_textbook_14q => (Class::QftTextbook, 14),
    qft_textbook_18q => (Class::QftTextbook, 18),
    random_10q => (Class::Random, 10),
    random_14q => (Class::Random, 14),
    random_18q => (Class::Random, 18),
    clifford_10q => (Class::Clifford, 10),
    clifford_14q => (Class::Clifford, 14),
    clifford_18q => (Class::Clifford, 18),
    clifford_t_10q => (Class::CliffordT, 10),
    clifford_t_14q => (Class::CliffordT, 14),
    clifford_t_18q => (Class::CliffordT, 18),
    rotations_10q => (Class::Rotations, 10),
    rotations_14q => (Class::Rotations, 14),
    rotations_18q => (Class::Rotations, 18),
}

// The sparse backend prunes entries with |a|^2 at or below 1e-16 after each gate and still
// reports the run exact. The opening rotation layer of an 18-qubit circuit is a product
// state whose smallest amplitudes are real and fall under that line (57 of them here), and
// later layers spread the loss: 9.9e-9 in amplitude, 3.7e-11 in probability. The unfused
// sparse run is off by 1.2e-8, while the statevector (fused and unfused), factored and
// tensor-network runs agree with each other to 4e-17. At 14 qubits no amplitude falls
// under the line and sparse agrees to 1e-16.
#[test]
#[ignore = "sparse default prune threshold drops real amplitude at 18 qubits"]
fn sparse_layered_mixed_18q_matches_statevector() {
    check_one(Participant::Sparse, Class::LayeredMixed, 18);
}

// Same mechanism, 34 amplitudes under the line: 2.3e-10 in amplitude, 3.6e-14 in
// probability.
#[test]
#[ignore = "sparse default prune threshold drops real amplitude at 18 qubits"]
fn sparse_matched_brickwork_18q_matches_statevector() {
    check_one(Participant::Sparse, Class::MatchedBrickwork, 18);
}

#[test]
fn every_case_compares_two_backends_and_every_backend_is_compared() {
    let mut used = Vec::new();
    for class in Class::ALL {
        for n in WIDTHS {
            let circuit = class.build(n);
            let (compared, skipped): (Vec<_>, Vec<_>) = Participant::ALL
                .into_iter()
                .map(|p| (p, p.skip(class, &circuit)))
                .partition(|(_, reason)| reason.is_none());
            let compared: Vec<Participant> = compared.into_iter().map(|(p, _)| p).collect();
            assert!(
                compared.len() >= 2,
                "{class:?} {n}q compares only {compared:?} against the statevector; \
                 skipped: {skipped:?}"
            );
            used.extend(compared);
        }
    }
    for participant in Participant::ALL {
        assert!(
            used.contains(&participant),
            "{participant:?} is skipped on every case"
        );
    }
}

// Each class is here to drive a fused form; a generator or threshold change that stops
// producing it would leave the class comparing a path nobody meant to test. The density
// matrix fuses its 10-qubit cases at the 20-qubit width of the buffer it sweeps.
#[test]
fn each_class_reaches_the_fused_form_it_targets() {
    let expected: &[(Class, usize, usize, &str)] = &[
        (Class::QuantumVolume, 14, 14, "multi_2q"),
        (Class::QuantumVolume, 18, 18, "multi_2q"),
        (Class::MatchedBrickwork, 18, 18, "multi_2q"),
        (Class::LayeredMixed, 14, 14, "fused_2q"),
        (Class::LayeredMixed, 18, 18, "multi_2q"),
        (Class::LayeredMixed, 18, 18, "diagonal_batch"),
        (Class::Hea, 14, 14, "multi_2q"),
        (Class::Qaoa, 14, 14, "multi_fused"),
        (Class::Qaoa, 18, 18, "batch_rzz"),
        (Class::DiagonalMixed, 18, 18, "diagonal_batch"),
        (Class::Qft, 18, 18, "qft_block"),
        (Class::QftTextbook, 18, 18, "batch_phase"),
        (Class::Clifford, 18, 18, "multi_2q"),
        (Class::Rotations, 14, 14, "multi_fused"),
        (Class::QuantumVolume, 10, 20, "multi_2q"),
        (Class::Qaoa, 10, 20, "batch_rzz"),
        (Class::Rotations, 10, 20, "multi_fused"),
    ];
    for &(class, n, fusion_width, gate_name) in expected {
        let circuit = class.build(n);
        let fused = fuse_circuit_for_width(&circuit, true, fusion_width);
        let found = fused
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Gate { gate, .. } if gate.name() == gate_name));
        assert!(
            found,
            "{class:?} {n}q fused at {fusion_width} qubits holds no {gate_name}"
        );
    }
}
