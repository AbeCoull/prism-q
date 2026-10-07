//! Cross-backend matrix tests over the shared small-circuit corpus.

mod common;

use common::circuits::{
    BackendKind, CircuitCase, exact_small_cases, find_case, product_separable_cases,
};
use common::{FACTORED_EPS, MPS_EPS, PRODUCT_EPS, SEED, SPARSE_EPS, STAB_EPS, TN_EPS};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::sim::{self, BackendKind as Kind};
use prism_q::simulate;

macro_rules! exact_backend_tests {
    ($suite:ident, $backend:expr, $constructor:expr, $eps:expr) => {
        $suite! {
            backend: $backend,
            constructor: $constructor,
            eps: $eps,
            cases: exact_small_cases(),
            coverage: corpus_is_complete,
            tests: {
                bell => "bell",
                ghz_3 => "ghz_3",
                ghz_4 => "ghz_4",
                ghz_5 => "ghz_5",
                qft_4 => "qft_4",
                qft_8 => "qft_8",
                random_4 => "random_4",
                random_8 => "random_8",
                hea_4 => "hea_4",
                qaoa_4 => "qaoa_4",
                qaoa_4_l3 => "qaoa_4_l3",
                qpe_4 => "qpe_4",
                qpe_8 => "qpe_8",
                cz_chain_8 => "cz_chain_8",
                w_state_4 => "w_state_4",
                single_qubit_rotations => "single_qubit_rotations",
                clifford_random_small => "clifford_random_small",
                sparse_basis_permutation => "sparse_basis_permutation",
            }
        }
    };
}

macro_rules! exact_backend {
    ($module:ident, $backend:expr, $constructor:expr, $eps:expr) => {
        mod $module {
            use super::*;

            exact_backend_tests!(backend_matrix_sv_tests, $backend, $constructor, $eps);
        }
    };
}

exact_backend!(
    sparse,
    BackendKind::Sparse,
    || SparseBackend::new(SEED),
    SPARSE_EPS
);
exact_backend!(mps, BackendKind::Mps, || MpsBackend::new(SEED, 64), MPS_EPS);
exact_backend!(
    tensor_network,
    BackendKind::TensorNetwork,
    || TensorNetworkBackend::new(SEED),
    TN_EPS
);
exact_backend!(
    factored,
    BackendKind::Factored,
    || FactoredBackend::new(SEED),
    FACTORED_EPS
);

mod sparse_fused {
    use super::*;

    exact_backend_tests!(
        backend_matrix_fused_tests,
        BackendKind::Sparse,
        || SparseBackend::new(SEED),
        SPARSE_EPS
    );
}

backend_matrix_sv_tests! {
    backend: BackendKind::Stabilizer,
    constructor: || StabilizerBackend::new(SEED),
    eps: STAB_EPS,
    cases: exact_small_cases(),
    coverage: matrix_stabilizer_corpus_is_complete,
    tests: {
        matrix_stabilizer_bell_matches_statevector => "bell",
        matrix_stabilizer_ghz_3_matches_statevector => "ghz_3",
        matrix_stabilizer_ghz_4_matches_statevector => "ghz_4",
        matrix_stabilizer_ghz_5_matches_statevector => "ghz_5",
        matrix_stabilizer_clifford_random_small_matches_statevector => "clifford_random_small",
        matrix_stabilizer_basis_permutation_matches_statevector => "sparse_basis_permutation",
    }
}

backend_matrix_sv_tests! {
    backend: BackendKind::Product,
    constructor: || ProductStateBackend::new(SEED),
    eps: PRODUCT_EPS,
    cases: product_separable_cases(),
    coverage: matrix_product_corpus_is_complete,
    tests: {
        matrix_product_single_qubit_rotations_4q_matches_statevector => "single_qubit_rotations_4q",
        matrix_product_single_qubit_rotations_8q_matches_statevector => "single_qubit_rotations_8q",
        matrix_product_single_qubit_rotations_12q_matches_statevector => "single_qubit_rotations_12q",
        matrix_product_single_qubit_rotations_16q_matches_statevector => "single_qubit_rotations_16q",
    }
}

backend_matrix_fused_tests! {
    backend: BackendKind::Product,
    constructor: || ProductStateBackend::new(SEED),
    eps: PRODUCT_EPS,
    cases: product_separable_cases(),
    coverage: matrix_product_fused_corpus_is_complete,
    tests: {
        matrix_product_single_qubit_rotations_4q_fused_matches_unfused => "single_qubit_rotations_4q",
        matrix_product_single_qubit_rotations_8q_fused_matches_unfused => "single_qubit_rotations_8q",
        matrix_product_single_qubit_rotations_12q_fused_matches_unfused => "single_qubit_rotations_12q",
        matrix_product_single_qubit_rotations_16q_fused_matches_unfused => "single_qubit_rotations_16q",
    }
}

fn run_on_new<B: Backend>(mut backend: B, circuit: &Circuit) -> B {
    sim::run_on(&mut backend, circuit).unwrap();
    backend
}

/// Include contiguous cuts and a scattered subsystem.
fn entropy_cuts(n: usize) -> Vec<Vec<usize>> {
    let mut cuts: Vec<Vec<usize>> = (1..n).map(|cut| (0..cut).collect()).collect();
    if n >= 4 {
        cuts.push((0..n).step_by(2).collect());
    }
    cuts
}

fn rdm_subsystems(n: usize) -> Vec<Vec<usize>> {
    let mut sets: Vec<Vec<usize>> = (1..=3.min(n)).map(|k| (0..k).collect()).collect();
    if n >= 4 {
        sets.push(vec![3, 1]);
    }
    sets
}

fn assert_entries(actual: &[Complex64], expected: &[Complex64], eps: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() < eps,
            "{label}: entry {i} reads {a} against {e}"
        );
    }
}

fn assert_diagnostics_match_statevector<B: Backend>(backend: &mut B, case: CircuitCase, eps: f64) {
    let circuit = case.circuit();
    let n = circuit.num_qubits;
    let mut sv = run_on_new(StatevectorBackend::new(SEED), &circuit);
    let name = case.name;
    for subsystem in entropy_cuts(n) {
        let want = sv.entanglement_entropy(&subsystem).unwrap();
        let got = backend.entanglement_entropy(&subsystem).unwrap();
        assert!(
            (got - want).abs() < eps,
            "{name}: entropy on {subsystem:?} reads {got} against {want}"
        );
    }
    for subsystem in rdm_subsystems(n) {
        let want = sv.reduced_density_matrix(&subsystem).unwrap();
        let got = backend.reduced_density_matrix(&subsystem).unwrap();
        assert_entries(
            &got,
            &want,
            eps,
            &format!("{name}: reduced density matrix on {subsystem:?}"),
        );
    }
}

// Tableau diagnostics provide an independent check against dense partial traces.
#[test]
fn matrix_stabilizer_diagnostics_match_the_statevector() {
    for case in exact_small_cases() {
        if !case.support(BackendKind::Stabilizer).is_supported() {
            continue;
        }
        let mut backend = run_on_new(StabilizerBackend::new(SEED), &case.circuit());
        assert_diagnostics_match_statevector(&mut backend, case, STAB_EPS);
    }
}

// Cuts across clusters exercise entropy addition and marginal tensor products.
#[test]
fn matrix_factored_stabilizer_diagnostics_match_the_statevector() {
    for case in exact_small_cases() {
        if !case.support(BackendKind::Stabilizer).is_supported() {
            continue;
        }
        let mut backend = run_on_new(FactoredStabilizerBackend::new(SEED), &case.circuit());
        assert_diagnostics_match_statevector(&mut backend, case, STAB_EPS);
    }
}

fn terminal_kind(backend: BackendKind) -> Kind {
    match backend {
        BackendKind::Sparse => Kind::Sparse,
        BackendKind::Mps => Kind::Mps { max_bond_dim: 64 },
        BackendKind::TensorNetwork => Kind::TensorNetwork,
        BackendKind::Factored => Kind::Factored,
        BackendKind::Stabilizer => Kind::Stabilizer,
        BackendKind::Product => Kind::ProductState,
    }
}

fn terminal_eps(backend: BackendKind) -> f64 {
    match backend {
        BackendKind::Sparse => SPARSE_EPS,
        BackendKind::Mps => MPS_EPS,
        BackendKind::TensorNetwork => TN_EPS,
        BackendKind::Factored => FACTORED_EPS,
        BackendKind::Stabilizer => STAB_EPS,
        BackendKind::Product => PRODUCT_EPS,
    }
}

fn overlap_through(
    left: Kind,
    left_circuit: &Circuit,
    right: Kind,
    right_circuit: &Circuit,
) -> f64 {
    simulate(left_circuit)
        .backend(left)
        .seed(SEED)
        .overlap(simulate(right_circuit).backend(right).seed(SEED))
        .unwrap()
        .fidelity
}

// A trailing Z makes overlap sensitive to phase errors on both query routes.
#[test]
fn matrix_overlap_agrees_with_the_statevector() {
    for case in exact_small_cases() {
        let circuit = case.circuit();
        let mut shifted = circuit.clone();
        shifted.add_gate(Gate::Z, &[0]);
        let name = case.name;
        let reference = overlap_through(Kind::Statevector, &circuit, Kind::Statevector, &shifted);
        for backend in [
            BackendKind::Sparse,
            BackendKind::Mps,
            BackendKind::TensorNetwork,
            BackendKind::Factored,
            BackendKind::Stabilizer,
            BackendKind::Product,
        ] {
            if !case.support(backend).is_supported() {
                continue;
            }
            let (kind, eps) = (terminal_kind(backend), terminal_eps(backend));
            let same = overlap_through(kind.clone(), &circuit, kind.clone(), &shifted);
            assert!(
                (same - reference).abs() < eps,
                "{name} on {}: pair overlap reads {same} against {reference}",
                backend.name()
            );
            let against_sv = overlap_through(kind, &circuit, Kind::Statevector, &circuit);
            assert!(
                (against_sv - 1.0).abs() < eps,
                "{name} on {}: overlap with its own statevector reads {against_sv}",
                backend.name()
            );
        }
    }
}

// Truncation discards norm; normalized self-overlap must remain one.
#[test]
fn matrix_truncated_mps_overlap_normalizes() {
    let circuit = find_case(exact_small_cases(), "random_8").circuit();
    let bounded = Kind::Mps { max_bond_dim: 2 };
    let fidelity = overlap_through(bounded.clone(), &circuit, bounded, &circuit);
    assert!((fidelity - 1.0).abs() < MPS_EPS, "fidelity {fidelity}");
}
