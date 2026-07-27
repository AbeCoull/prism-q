#![allow(dead_code)]

use prism_q::backend::Backend;

use super::circuits::{BackendKind, CircuitCase};
use super::{
    assert_backend_matches_sv, assert_backend_outcome_matches_sv, assert_backend_repeatable,
    assert_fused_matches_unfused,
};

/// A backend the corpus marks as rejected for this case must not reach a
/// comparison: the skip belongs in the case list, not in a silent pass here.
fn assert_supported(backend_kind: BackendKind, case: CircuitCase) {
    assert!(
        case.support(backend_kind).is_supported(),
        "{} {} is marked rejected in the shared corpus",
        backend_kind.name(),
        case.name
    );
}

pub fn assert_backend_case_matches_sv<B, F>(
    backend_kind: BackendKind,
    case: CircuitCase,
    new_backend: F,
    eps: f64,
) where
    B: Backend,
    F: Fn() -> B,
{
    assert_supported(backend_kind, case);
    let circuit = case.circuit();
    let label = format!("{} {}", backend_kind.name(), case.name);
    let mut backend = new_backend();
    assert_backend_matches_sv(&mut backend, &circuit, eps, &label);
}

pub fn assert_backend_matrix_matches_sv<B, F>(
    backend_kind: BackendKind,
    cases: &[CircuitCase],
    new_backend: F,
    eps: f64,
) where
    B: Backend,
    F: Fn() -> B,
{
    for case in cases {
        assert_backend_case_matches_sv(backend_kind, *case, &new_backend, eps);
    }
}

pub fn assert_backend_case_outcome_matches_sv<B, F>(
    backend_kind: BackendKind,
    case: CircuitCase,
    new_backend: F,
    eps: f64,
) where
    B: Backend,
    F: Fn() -> B,
{
    assert_supported(backend_kind, case);
    let circuit = case.circuit();
    let label = format!("{} {}", backend_kind.name(), case.name);
    let mut backend = new_backend();
    assert_backend_outcome_matches_sv(&mut backend, &circuit, eps, &label);
}

pub fn assert_backend_case_repeatable<B, F>(
    backend_kind: BackendKind,
    case: CircuitCase,
    new_backend: F,
    eps: f64,
) where
    B: Backend,
    F: Fn() -> B,
{
    assert_supported(backend_kind, case);
    let circuit = case.circuit();
    let label = format!("{} {} repeatable", backend_kind.name(), case.name);
    assert_backend_repeatable(new_backend, &circuit, eps, &label);
}

pub fn assert_backend_case_fused_matches_unfused<B, F>(
    backend_kind: BackendKind,
    case: CircuitCase,
    new_backend: F,
    eps: f64,
) where
    B: Backend,
    F: Fn() -> B,
{
    assert_supported(backend_kind, case);
    let circuit = case.circuit();
    let label = format!("{} {} fused", backend_kind.name(), case.name);
    assert_fused_matches_unfused(&new_backend, &circuit, eps, &label);
}

pub fn assert_backend_matrix_fused_matches_unfused<B, F>(
    backend_kind: BackendKind,
    cases: &[CircuitCase],
    new_backend: F,
    eps: f64,
) where
    B: Backend,
    F: Fn() -> B,
{
    for case in cases {
        assert_backend_case_fused_matches_unfused(backend_kind, *case, &new_backend, eps);
    }
}
