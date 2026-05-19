#![allow(dead_code)]

use prism_q::backend::Backend;

use prism_q::circuit::Circuit;

use super::circuits::{BackendKind, CircuitCase};
use super::{
    assert_backend_matches_sv, assert_backend_outcome_matches_sv, assert_backend_repeatable,
    assert_fused_matches_unfused,
};

fn checked_eps(
    backend_kind: BackendKind,
    case: CircuitCase,
    circuit: &Circuit,
    fallback: f64,
) -> f64 {
    let expectation = case.expectation(backend_kind);
    assert!(
        expectation.is_supported(),
        "{} {} is marked rejected in the shared corpus",
        backend_kind.name(),
        case.name
    );
    if let Some(max_qubits) = expectation.max_qubits {
        assert!(
            circuit.num_qubits <= max_qubits,
            "{} {} uses {} qubits, above the shared corpus limit of {}",
            backend_kind.name(),
            case.name,
            circuit.num_qubits,
            max_qubits
        );
    }
    assert!(
        !expectation.export_required,
        "{} {} requires an export-aware matrix helper",
        backend_kind.name(),
        case.name
    );
    expectation.tolerance_override.unwrap_or(fallback)
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
    let circuit = case.circuit();
    let eps = checked_eps(backend_kind, case, &circuit, eps);
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
    let circuit = case.circuit();
    let eps = checked_eps(backend_kind, case, &circuit, eps);
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
    let circuit = case.circuit();
    let eps = checked_eps(backend_kind, case, &circuit, eps);
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
    let circuit = case.circuit();
    let eps = checked_eps(backend_kind, case, &circuit, eps);
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
