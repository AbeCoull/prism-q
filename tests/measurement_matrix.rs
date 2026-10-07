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

macro_rules! measurement_tests {
    (
        backend: $backend:expr,
        constructor: $constructor:expr,
        eps: $eps:expr,
        regions: { $($region:ident => $region_case:literal),* $(,)? },
        parity: { $($parity:ident => $parity_case:literal),* $(,)? }
    ) => {
        backend_matrix_outcome_tests! {
            backend: $backend,
            constructor: $constructor,
            eps: $eps,
            cases: measurement_cases(),
            coverage: measurement_corpus_is_complete,
            tests: {
                deterministic => "deterministic_measurement",
                reset_from_one => "reset_from_one",
                reset_from_one_with_spectator => "reset_from_one_with_spectator",
                reset_conditional => "measurement_reset_conditional",
                $($region => $region_case,)*
            }
        }

        backend_matrix_repeatability_tests! {
            backend: $backend,
            constructor: $constructor,
            eps: $eps,
            cases: random_measurement_cases(),
            coverage: repeatability_corpus_is_complete,
            tests: {
                superposition_repeatable => "superposition_measurement",
            }
        }

        backend_matrix_outcome_tests! {
            backend: $backend,
            constructor: $constructor,
            eps: $eps,
            cases: pauli_measurement_cases(),
            coverage: pauli_corpus_is_complete,
            tests: {
                measure_x_basis_on_plus => "measure_x_basis_on_plus",
                measure_y_basis_on_plus_i => "measure_y_basis_on_plus_i",
                $($parity => $parity_case,)*
            }
        }
    };
}

macro_rules! entangling_measurement_tests {
    ($module:ident, $backend:expr, $constructor:expr, $eps:expr) => {
        mod $module {
            use super::*;

            measurement_tests! {
                backend: $backend,
                constructor: $constructor,
                eps: $eps,
                regions: {
                    reset_region => "measurement_reset_region",
                    parity_sibling_regions => "parity_sibling_regions",
                },
                parity: {
                    parity_zz_on_bell => "parity_zz_on_bell",
                    parity_xx_on_plus_plus => "parity_xx_on_plus_plus",
                    parity_yy_on_bell => "parity_yy_on_bell",
                    parity_xz_on_plus_one => "parity_xz_on_plus_one",
                    parity_xzxz_weight_4 => "parity_xzxz_weight_4",
                }
            }
        }
    };
}

entangling_measurement_tests!(
    sparse,
    BackendKind::Sparse,
    || SparseBackend::new(SEED),
    SPARSE_EPS
);
entangling_measurement_tests!(mps, BackendKind::Mps, || MpsBackend::new(SEED, 64), MPS_EPS);
entangling_measurement_tests!(
    tensor_network,
    BackendKind::TensorNetwork,
    || TensorNetworkBackend::new(SEED),
    TN_EPS
);
entangling_measurement_tests!(
    factored,
    BackendKind::Factored,
    || FactoredBackend::new(SEED),
    FACTORED_EPS
);
entangling_measurement_tests!(
    stabilizer,
    BackendKind::Stabilizer,
    || StabilizerBackend::new(SEED),
    STAB_EPS
);

mod product {
    use super::*;

    measurement_tests! {
        backend: BackendKind::Product,
        constructor: || ProductStateBackend::new(SEED),
        eps: PRODUCT_EPS,
        regions: {},
        parity: {}
    }
}
