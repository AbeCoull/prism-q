//! Cross-backend matrix tests over the shared small-circuit corpus.

mod common;

use common::circuits::{BackendKind, exact_small_cases, product_separable_cases};
use common::{FACTORED_EPS, MPS_EPS, PRODUCT_EPS, SEED, SPARSE_EPS, STAB_EPS, TN_EPS};
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;

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
