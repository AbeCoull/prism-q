//! Measurement, reset, conditional, and seed repeatability matrix tests.

mod common;

use common::circuits::{
    BackendKind, measurement_cases, pauli_measurement_cases, random_measurement_cases,
};
use common::{FACTORED_EPS, MPS_EPS, PRODUCT_EPS, SEED, SPARSE_EPS, STAB_EPS, TN_EPS};
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;

backend_matrix_outcome_tests! {
    backend: BackendKind::Sparse,
    constructor: || SparseBackend::new(SEED),
    eps: SPARSE_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_sparse_deterministic_matches_statevector => "deterministic_measurement",
        measurement_sparse_reset_from_one_matches_statevector => "reset_from_one",
        measurement_sparse_reset_from_one_with_spectator_matches_statevector => "reset_from_one_with_spectator",
        measurement_sparse_reset_conditional_matches_statevector => "measurement_reset_conditional",
        measurement_sparse_reset_region_matches_statevector => "measurement_reset_region",
    }
}

backend_matrix_repeatability_tests! {
    backend: BackendKind::Sparse,
    constructor: || SparseBackend::new(SEED),
    eps: SPARSE_EPS,
    cases: random_measurement_cases(),
    tests: {
        measurement_sparse_superposition_repeatable => "superposition_measurement",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Sparse,
    constructor: || SparseBackend::new(SEED),
    eps: SPARSE_EPS,
    cases: pauli_measurement_cases(),
    tests: {
        measurement_sparse_measure_x_basis_on_plus_matches_statevector => "measure_x_basis_on_plus",
        measurement_sparse_measure_y_basis_on_plus_i_matches_statevector => "measure_y_basis_on_plus_i",
        measurement_sparse_parity_zz_on_bell_matches_statevector => "parity_zz_on_bell",
        measurement_sparse_parity_xx_on_plus_plus_matches_statevector => "parity_xx_on_plus_plus",
        measurement_sparse_parity_yy_on_bell_matches_statevector => "parity_yy_on_bell",
        measurement_sparse_parity_xz_on_plus_one_matches_statevector => "parity_xz_on_plus_one",
        measurement_sparse_parity_xzxz_weight_4_matches_statevector => "parity_xzxz_weight_4",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Mps,
    constructor: || MpsBackend::new(SEED, 64),
    eps: MPS_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_mps_deterministic_matches_statevector => "deterministic_measurement",
        measurement_mps_reset_from_one_matches_statevector => "reset_from_one",
        measurement_mps_reset_from_one_with_spectator_matches_statevector => "reset_from_one_with_spectator",
        measurement_mps_reset_conditional_matches_statevector => "measurement_reset_conditional",
        measurement_mps_reset_region_matches_statevector => "measurement_reset_region",
    }
}

backend_matrix_repeatability_tests! {
    backend: BackendKind::Mps,
    constructor: || MpsBackend::new(SEED, 64),
    eps: MPS_EPS,
    cases: random_measurement_cases(),
    tests: {
        measurement_mps_superposition_repeatable => "superposition_measurement",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Mps,
    constructor: || MpsBackend::new(SEED, 64),
    eps: MPS_EPS,
    cases: pauli_measurement_cases(),
    tests: {
        measurement_mps_measure_x_basis_on_plus_matches_statevector => "measure_x_basis_on_plus",
        measurement_mps_measure_y_basis_on_plus_i_matches_statevector => "measure_y_basis_on_plus_i",
        measurement_mps_parity_zz_on_bell_matches_statevector => "parity_zz_on_bell",
        measurement_mps_parity_xx_on_plus_plus_matches_statevector => "parity_xx_on_plus_plus",
        measurement_mps_parity_yy_on_bell_matches_statevector => "parity_yy_on_bell",
        measurement_mps_parity_xz_on_plus_one_matches_statevector => "parity_xz_on_plus_one",
        measurement_mps_parity_xzxz_weight_4_matches_statevector => "parity_xzxz_weight_4",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::TensorNetwork,
    constructor: || TensorNetworkBackend::new(SEED),
    eps: TN_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_tensor_network_deterministic_matches_statevector => "deterministic_measurement",
        measurement_tensor_network_reset_from_one_matches_statevector => "reset_from_one",
        measurement_tensor_network_reset_from_one_with_spectator_matches_statevector => "reset_from_one_with_spectator",
        measurement_tensor_network_reset_conditional_matches_statevector => "measurement_reset_conditional",
        measurement_tensor_network_reset_region_matches_statevector => "measurement_reset_region",
    }
}

backend_matrix_repeatability_tests! {
    backend: BackendKind::TensorNetwork,
    constructor: || TensorNetworkBackend::new(SEED),
    eps: TN_EPS,
    cases: random_measurement_cases(),
    tests: {
        measurement_tensor_network_superposition_repeatable => "superposition_measurement",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::TensorNetwork,
    constructor: || TensorNetworkBackend::new(SEED),
    eps: TN_EPS,
    cases: pauli_measurement_cases(),
    tests: {
        measurement_tensor_network_measure_x_basis_on_plus_matches_statevector => "measure_x_basis_on_plus",
        measurement_tensor_network_measure_y_basis_on_plus_i_matches_statevector => "measure_y_basis_on_plus_i",
        measurement_tensor_network_parity_zz_on_bell_matches_statevector => "parity_zz_on_bell",
        measurement_tensor_network_parity_xx_on_plus_plus_matches_statevector => "parity_xx_on_plus_plus",
        measurement_tensor_network_parity_yy_on_bell_matches_statevector => "parity_yy_on_bell",
        measurement_tensor_network_parity_xz_on_plus_one_matches_statevector => "parity_xz_on_plus_one",
        measurement_tensor_network_parity_xzxz_weight_4_matches_statevector => "parity_xzxz_weight_4",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Factored,
    constructor: || FactoredBackend::new(SEED),
    eps: FACTORED_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_factored_deterministic_matches_statevector => "deterministic_measurement",
        measurement_factored_reset_from_one_matches_statevector => "reset_from_one",
        measurement_factored_reset_from_one_with_spectator_matches_statevector => "reset_from_one_with_spectator",
        measurement_factored_reset_conditional_matches_statevector => "measurement_reset_conditional",
        measurement_factored_reset_region_matches_statevector => "measurement_reset_region",
    }
}

backend_matrix_repeatability_tests! {
    backend: BackendKind::Factored,
    constructor: || FactoredBackend::new(SEED),
    eps: FACTORED_EPS,
    cases: random_measurement_cases(),
    tests: {
        measurement_factored_superposition_repeatable => "superposition_measurement",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Factored,
    constructor: || FactoredBackend::new(SEED),
    eps: FACTORED_EPS,
    cases: pauli_measurement_cases(),
    tests: {
        measurement_factored_measure_x_basis_on_plus_matches_statevector => "measure_x_basis_on_plus",
        measurement_factored_measure_y_basis_on_plus_i_matches_statevector => "measure_y_basis_on_plus_i",
        measurement_factored_parity_zz_on_bell_matches_statevector => "parity_zz_on_bell",
        measurement_factored_parity_xx_on_plus_plus_matches_statevector => "parity_xx_on_plus_plus",
        measurement_factored_parity_yy_on_bell_matches_statevector => "parity_yy_on_bell",
        measurement_factored_parity_xz_on_plus_one_matches_statevector => "parity_xz_on_plus_one",
        measurement_factored_parity_xzxz_weight_4_matches_statevector => "parity_xzxz_weight_4",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Stabilizer,
    constructor: || StabilizerBackend::new(SEED),
    eps: STAB_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_stabilizer_deterministic_matches_statevector => "deterministic_measurement",
        measurement_stabilizer_reset_from_one_matches_statevector => "reset_from_one",
        measurement_stabilizer_reset_from_one_with_spectator_matches_statevector => "reset_from_one_with_spectator",
        measurement_stabilizer_reset_conditional_matches_statevector => "measurement_reset_conditional",
        measurement_stabilizer_reset_region_matches_statevector => "measurement_reset_region",
    }
}

backend_matrix_repeatability_tests! {
    backend: BackendKind::Stabilizer,
    constructor: || StabilizerBackend::new(SEED),
    eps: STAB_EPS,
    cases: random_measurement_cases(),
    tests: {
        measurement_stabilizer_superposition_repeatable => "superposition_measurement",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Stabilizer,
    constructor: || StabilizerBackend::new(SEED),
    eps: STAB_EPS,
    cases: pauli_measurement_cases(),
    tests: {
        measurement_stabilizer_measure_x_basis_on_plus_matches_statevector => "measure_x_basis_on_plus",
        measurement_stabilizer_measure_y_basis_on_plus_i_matches_statevector => "measure_y_basis_on_plus_i",
        measurement_stabilizer_parity_zz_on_bell_matches_statevector => "parity_zz_on_bell",
        measurement_stabilizer_parity_xx_on_plus_plus_matches_statevector => "parity_xx_on_plus_plus",
        measurement_stabilizer_parity_yy_on_bell_matches_statevector => "parity_yy_on_bell",
        measurement_stabilizer_parity_xz_on_plus_one_matches_statevector => "parity_xz_on_plus_one",
        measurement_stabilizer_parity_xzxz_weight_4_matches_statevector => "parity_xzxz_weight_4",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Product,
    constructor: || ProductStateBackend::new(SEED),
    eps: PRODUCT_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_product_deterministic_matches_statevector => "deterministic_measurement",
        measurement_product_reset_from_one_matches_statevector => "reset_from_one",
        measurement_product_reset_from_one_with_spectator_matches_statevector => "reset_from_one_with_spectator",
        measurement_product_reset_conditional_matches_statevector => "measurement_reset_conditional",
    }
}

backend_matrix_repeatability_tests! {
    backend: BackendKind::Product,
    constructor: || ProductStateBackend::new(SEED),
    eps: PRODUCT_EPS,
    cases: random_measurement_cases(),
    tests: {
        measurement_product_superposition_repeatable => "superposition_measurement",
    }
}

backend_matrix_outcome_tests! {
    backend: BackendKind::Product,
    constructor: || ProductStateBackend::new(SEED),
    eps: PRODUCT_EPS,
    cases: pauli_measurement_cases(),
    tests: {
        measurement_product_measure_x_basis_on_plus_matches_statevector => "measure_x_basis_on_plus",
        measurement_product_measure_y_basis_on_plus_i_matches_statevector => "measure_y_basis_on_plus_i",
    }
}
