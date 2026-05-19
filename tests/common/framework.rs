#[macro_export]
macro_rules! circuit_case_tests {
    (
        cases: $cases:expr,
        runner: $runner:path,
        tests: {
            $($test_name:ident => $case_name:literal),+ $(,)?
        }
    ) => {
        $(
            #[test]
            fn $test_name() {
                let case = $crate::common::circuits::find_case($cases, $case_name);
                let circuit = case.circuit();
                $runner(&circuit);
            }
        )+
    };
}

#[macro_export]
macro_rules! backend_matrix_sv_tests {
    (
        backend: $backend_kind:expr,
        constructor: $new_backend:expr,
        eps: $eps:expr,
        cases: $cases:expr,
        tests: {
            $($test_name:ident => $case_name:literal),+ $(,)?
        }
    ) => {
        $(
            #[test]
            fn $test_name() {
                let case = $crate::common::circuits::find_case($cases, $case_name);
                $crate::common::matrix::assert_backend_case_matches_sv(
                    $backend_kind,
                    case,
                    $new_backend,
                    $eps,
                );
            }
        )+
    };
}

#[macro_export]
macro_rules! backend_matrix_fused_tests {
    (
        backend: $backend_kind:expr,
        constructor: $new_backend:expr,
        eps: $eps:expr,
        cases: $cases:expr,
        tests: {
            $($test_name:ident => $case_name:literal),+ $(,)?
        }
    ) => {
        $(
            #[test]
            fn $test_name() {
                let case = $crate::common::circuits::find_case($cases, $case_name);
                $crate::common::matrix::assert_backend_case_fused_matches_unfused(
                    $backend_kind,
                    case,
                    $new_backend,
                    $eps,
                );
            }
        )+
    };
}

#[macro_export]
macro_rules! backend_matrix_outcome_tests {
    (
        backend: $backend_kind:expr,
        constructor: $new_backend:expr,
        eps: $eps:expr,
        cases: $cases:expr,
        tests: {
            $($test_name:ident => $case_name:literal),+ $(,)?
        }
    ) => {
        $(
            #[test]
            fn $test_name() {
                let case = $crate::common::circuits::find_case($cases, $case_name);
                $crate::common::matrix::assert_backend_case_outcome_matches_sv(
                    $backend_kind,
                    case,
                    $new_backend,
                    $eps,
                );
            }
        )+
    };
}

#[macro_export]
macro_rules! backend_matrix_repeatability_tests {
    (
        backend: $backend_kind:expr,
        constructor: $new_backend:expr,
        eps: $eps:expr,
        cases: $cases:expr,
        tests: {
            $($test_name:ident => $case_name:literal),+ $(,)?
        }
    ) => {
        $(
            #[test]
            fn $test_name() {
                let case = $crate::common::circuits::find_case($cases, $case_name);
                $crate::common::matrix::assert_backend_case_repeatable(
                    $backend_kind,
                    case,
                    $new_backend,
                    $eps,
                );
            }
        )+
    };
}
