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

backend_matrix_sv_tests! {
    backend: BackendKind::Sparse,
    constructor: || SparseBackend::new(SEED),
    eps: SPARSE_EPS,
    cases: exact_small_cases(),
    tests: {
        matrix_sparse_bell_matches_statevector => "bell",
        matrix_sparse_ghz_3_matches_statevector => "ghz_3",
        matrix_sparse_ghz_5_matches_statevector => "ghz_5",
        matrix_sparse_qft_4_matches_statevector => "qft_4",
        matrix_sparse_qft_8_matches_statevector => "qft_8",
        matrix_sparse_random_4_matches_statevector => "random_4",
        matrix_sparse_random_8_matches_statevector => "random_8",
        matrix_sparse_hea_4_matches_statevector => "hea_4",
        matrix_sparse_ghz_4_matches_statevector => "ghz_4",
        matrix_sparse_qaoa_4_l3_matches_statevector => "qaoa_4_l3",
        matrix_sparse_qpe_4_matches_statevector => "qpe_4",
        matrix_sparse_qpe_8_matches_statevector => "qpe_8",
        matrix_sparse_cz_chain_8_matches_statevector => "cz_chain_8",
        matrix_sparse_single_qubit_rotations_matches_statevector => "single_qubit_rotations",
        matrix_sparse_clifford_random_small_matches_statevector => "clifford_random_small",
        matrix_sparse_basis_permutation_matches_statevector => "sparse_basis_permutation",
    }
}

backend_matrix_fused_tests! {
    backend: BackendKind::Sparse,
    constructor: || SparseBackend::new(SEED),
    eps: SPARSE_EPS,
    cases: exact_small_cases(),
    tests: {
        matrix_sparse_bell_fused_matches_unfused => "bell",
        matrix_sparse_ghz_3_fused_matches_unfused => "ghz_3",
        matrix_sparse_ghz_5_fused_matches_unfused => "ghz_5",
        matrix_sparse_qft_4_fused_matches_unfused => "qft_4",
        matrix_sparse_single_qubit_rotations_fused_matches_unfused => "single_qubit_rotations",
        matrix_sparse_clifford_random_small_fused_matches_unfused => "clifford_random_small",
        matrix_sparse_basis_permutation_fused_matches_unfused => "sparse_basis_permutation",
    }
}

backend_matrix_sv_tests! {
    backend: BackendKind::Mps,
    constructor: || MpsBackend::new(SEED, 64),
    eps: MPS_EPS,
    cases: exact_small_cases(),
    tests: {
        matrix_mps_bell_matches_statevector => "bell",
        matrix_mps_ghz_3_matches_statevector => "ghz_3",
        matrix_mps_ghz_5_matches_statevector => "ghz_5",
        matrix_mps_qft_4_matches_statevector => "qft_4",
        matrix_mps_qft_8_matches_statevector => "qft_8",
        matrix_mps_random_4_matches_statevector => "random_4",
        matrix_mps_random_8_matches_statevector => "random_8",
        matrix_mps_hea_4_matches_statevector => "hea_4",
        matrix_mps_ghz_4_matches_statevector => "ghz_4",
        matrix_mps_qaoa_4_matches_statevector => "qaoa_4",
        matrix_mps_qpe_4_matches_statevector => "qpe_4",
        matrix_mps_qpe_8_matches_statevector => "qpe_8",
        matrix_mps_w_state_4_matches_statevector => "w_state_4",
        matrix_mps_single_qubit_rotations_matches_statevector => "single_qubit_rotations",
        matrix_mps_clifford_random_small_matches_statevector => "clifford_random_small",
        matrix_mps_basis_permutation_matches_statevector => "sparse_basis_permutation",
    }
}

backend_matrix_sv_tests! {
    backend: BackendKind::TensorNetwork,
    constructor: || TensorNetworkBackend::new(SEED),
    eps: TN_EPS,
    cases: exact_small_cases(),
    tests: {
        matrix_tensor_network_bell_matches_statevector => "bell",
        matrix_tensor_network_ghz_3_matches_statevector => "ghz_3",
        matrix_tensor_network_ghz_5_matches_statevector => "ghz_5",
        matrix_tensor_network_qft_4_matches_statevector => "qft_4",
        matrix_tensor_network_qft_8_matches_statevector => "qft_8",
        matrix_tensor_network_random_4_matches_statevector => "random_4",
        matrix_tensor_network_random_8_matches_statevector => "random_8",
        matrix_tensor_network_hea_4_matches_statevector => "hea_4",
        matrix_tensor_network_ghz_4_matches_statevector => "ghz_4",
        matrix_tensor_network_qaoa_4_matches_statevector => "qaoa_4",
        matrix_tensor_network_qpe_4_matches_statevector => "qpe_4",
        matrix_tensor_network_qpe_8_matches_statevector => "qpe_8",
        matrix_tensor_network_cz_chain_8_matches_statevector => "cz_chain_8",
        matrix_tensor_network_w_state_4_matches_statevector => "w_state_4",
        matrix_tensor_network_single_qubit_rotations_matches_statevector => "single_qubit_rotations",
        matrix_tensor_network_clifford_random_small_matches_statevector => "clifford_random_small",
        matrix_tensor_network_basis_permutation_matches_statevector => "sparse_basis_permutation",
    }
}

backend_matrix_sv_tests! {
    backend: BackendKind::Factored,
    constructor: || FactoredBackend::new(SEED),
    eps: FACTORED_EPS,
    cases: exact_small_cases(),
    tests: {
        matrix_factored_bell_matches_statevector => "bell",
        matrix_factored_ghz_3_matches_statevector => "ghz_3",
        matrix_factored_ghz_5_matches_statevector => "ghz_5",
        matrix_factored_qft_4_matches_statevector => "qft_4",
        matrix_factored_single_qubit_rotations_matches_statevector => "single_qubit_rotations",
        matrix_factored_clifford_random_small_matches_statevector => "clifford_random_small",
        matrix_factored_basis_permutation_matches_statevector => "sparse_basis_permutation",
    }
}

backend_matrix_sv_tests! {
    backend: BackendKind::Stabilizer,
    constructor: || StabilizerBackend::new(SEED),
    eps: STAB_EPS,
    cases: exact_small_cases(),
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
    tests: {
        matrix_product_single_qubit_rotations_4q_fused_matches_unfused => "single_qubit_rotations_4q",
        matrix_product_single_qubit_rotations_8q_fused_matches_unfused => "single_qubit_rotations_8q",
        matrix_product_single_qubit_rotations_12q_fused_matches_unfused => "single_qubit_rotations_12q",
        matrix_product_single_qubit_rotations_16q_fused_matches_unfused => "single_qubit_rotations_16q",
    }
}

// ---- State diagnostics ----

fn run_on_new<B: Backend>(mut backend: B, circuit: &Circuit) -> B {
    sim::run_on(&mut backend, circuit).unwrap();
    backend
}

/// Every contiguous cut of an `n`-qubit register plus one scattered
/// subsystem, so a backend that only handles a prefix is caught.
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

// The entropy comes off a rank and the marginal off a projector, neither of
// which touches an amplitude, so both are checked against the dense partial
// trace on every Clifford case of the corpus.
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

// Clusters are unentangled, so the entropy sums over them and the marginal is
// their Kronecker product; a cut that crosses several clusters exercises both.
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
        BackendKind::TensorNetwork => Kind::TensorNetwork { tolerance: None },
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

// Each backend against itself on a pair of circuits, which takes its native
// route, and against the statevector on one circuit, which takes the dense
// export. The second circuit differs by a trailing Z, so the value is neither
// 1 nor 0 on most cases and a dropped phase would show.
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

// A chain at bond 2 discards weight on `random_8`, so the state it holds is
// unnormalized. The overlap divides by both norms, which is what leaves a
// truncated chain reading 1 against itself.
#[test]
fn matrix_truncated_mps_overlap_normalizes() {
    let circuit = find_case(exact_small_cases(), "random_8").circuit();
    let bounded = Kind::Mps { max_bond_dim: 2 };
    let fidelity = overlap_through(bounded.clone(), &circuit, bounded, &circuit);
    assert!((fidelity - 1.0).abs() < MPS_EPS, "fidelity {fidelity}");
}
