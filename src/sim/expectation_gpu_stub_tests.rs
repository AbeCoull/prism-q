use super::unified_pauli::{PauliAxis, PauliTerm};
use super::*;
use crate::gates::Gate;
use crate::gpu::GpuContext;

fn dense_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Rx(0.3), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

fn observables() -> Vec<Vec<PauliTerm>> {
    vec![
        vec![PauliTerm::new(0, PauliAxis::Z)],
        vec![
            PauliTerm::new(1, PauliAxis::X),
            PauliTerm::new(2, PauliAxis::Y),
        ],
    ]
}

/// The two routes run the same CPU kernels but as separate invocations, so
/// a parallel reduction can pair its partial sums differently and land a
/// ulp apart. Agreement is what the test is about, not bit equality.
fn assert_expectations_close(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (slot, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "observable {slot}: {a:.17} vs {b:.17}"
        );
    }
}

#[test]
fn auto_gpu_expectation_matches_auto_on_stub() {
    let circuit = dense_circuit(16);
    let auto_vals =
        run_expectation_values_with(BackendKind::Auto, &circuit, &observables(), 42).unwrap();
    let gpu_vals = run_expectation_values_with(
        BackendKind::AutoGpu {
            context: GpuContext::stub_for_tests(),
        },
        &circuit,
        &observables(),
        42,
    )
    .unwrap();
    assert_expectations_close(&auto_vals, &gpu_vals);
}

// Explicit `StatevectorGpu` expectation values resolve hard above the
// crossover, so the stub's failed allocation surfaces.
#[test]
fn statevector_gpu_expectation_hard_above_crossover_on_stub() {
    let circuit = dense_circuit(16);
    let err = run_expectation_values_with(
        BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        },
        &circuit,
        &observables(),
        42,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        crate::error::PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn statevector_gpu_expectation_below_crossover_matches_statevector() {
    let circuit = dense_circuit(6);
    let sv = run_expectation_values_with(BackendKind::Statevector, &circuit, &observables(), 42)
        .unwrap();
    let gpu = run_expectation_values_with(
        BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        },
        &circuit,
        &observables(),
        42,
    )
    .unwrap();
    assert_expectations_close(&sv, &gpu);
}
