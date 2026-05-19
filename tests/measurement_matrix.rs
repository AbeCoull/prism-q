//! Measurement, reset, conditional, and seed repeatability matrix tests.

mod common;

use common::circuits::{measurement_cases, random_measurement_cases, BackendKind};
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
        measurement_sparse_reset_conditional_matches_statevector => "measurement_reset_conditional",
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
    backend: BackendKind::Mps,
    constructor: || MpsBackend::new(SEED, 64),
    eps: MPS_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_mps_deterministic_matches_statevector => "deterministic_measurement",
        measurement_mps_reset_from_one_matches_statevector => "reset_from_one",
        measurement_mps_reset_conditional_matches_statevector => "measurement_reset_conditional",
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
    backend: BackendKind::TensorNetwork,
    constructor: || TensorNetworkBackend::new(SEED),
    eps: TN_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_tensor_network_deterministic_matches_statevector => "deterministic_measurement",
        measurement_tensor_network_reset_from_one_matches_statevector => "reset_from_one",
        measurement_tensor_network_reset_conditional_matches_statevector => "measurement_reset_conditional",
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
    backend: BackendKind::Factored,
    constructor: || FactoredBackend::new(SEED),
    eps: FACTORED_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_factored_deterministic_matches_statevector => "deterministic_measurement",
        measurement_factored_reset_from_one_matches_statevector => "reset_from_one",
        measurement_factored_reset_conditional_matches_statevector => "measurement_reset_conditional",
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
    backend: BackendKind::Stabilizer,
    constructor: || StabilizerBackend::new(SEED),
    eps: STAB_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_stabilizer_deterministic_matches_statevector => "deterministic_measurement",
        measurement_stabilizer_reset_from_one_matches_statevector => "reset_from_one",
        measurement_stabilizer_reset_conditional_matches_statevector => "measurement_reset_conditional",
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
    backend: BackendKind::Product,
    constructor: || ProductStateBackend::new(SEED),
    eps: PRODUCT_EPS,
    cases: measurement_cases(),
    tests: {
        measurement_product_deterministic_matches_statevector => "deterministic_measurement",
        measurement_product_reset_from_one_matches_statevector => "reset_from_one",
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
