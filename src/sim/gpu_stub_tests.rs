use super::terminal_candidate_matrix_tests::rx_cx_chain;
use super::unified_pauli::{PauliAxis, PauliTerm};
use super::*;
use crate::gates::Gate;
use crate::gpu::GpuContext;

fn chain_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    rx_cx_chain(&mut c, 0..n);
    c
}

fn measured_chain_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, n);
    rx_cx_chain(&mut c, 0..n);
    for q in 0..n {
        c.add_measure(q, q);
    }
    c
}

fn auto_gpu_stub() -> BackendKind {
    BackendKind::AutoGpu {
        context: GpuContext::stub_for_tests(),
    }
}

fn statevector_gpu_stub() -> BackendKind {
    BackendKind::StatevectorGpu {
        context: GpuContext::stub_for_tests(),
    }
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

// The terminal fast path resolves the accel through the capability table.
// On the stub the VRAM gate fails closed, so `AutoGpu` must take the
// identical host path as `Auto`: same sampler, same RNG stream, byte-equal
// counts.
#[test]
fn auto_gpu_terminal_counts_match_auto_on_stub() {
    let circuit = measured_chain_circuit(16);
    let auto_counts = run_counts_with(BackendKind::Auto, &circuit, 500, 42)
        .unwrap()
        .0;
    let gpu_counts = run_counts_with(auto_gpu_stub(), &circuit, 500, 42)
        .unwrap()
        .0;
    assert_eq!(auto_counts, gpu_counts);
}

#[test]
fn auto_gpu_terminal_shots_match_auto_on_stub() {
    let circuit = measured_chain_circuit(16);
    let auto_shots = run_shots_with(BackendKind::Auto, &circuit, 64, 42).unwrap();
    let gpu_shots = run_shots_with(auto_gpu_stub(), &circuit, 64, 42).unwrap();
    assert_eq!(auto_shots.shots, gpu_shots.shots);
}

// Explicit `StatevectorGpu` is a terminal-fast-path candidate. Above the
// crossover it resolves hard, so the stub's failed allocation surfaces
// instead of falling back, proving the device path was reached.
#[test]
fn statevector_gpu_terminal_counts_hard_above_crossover_on_stub() {
    let circuit = measured_chain_circuit(16);
    let err = run_counts_with(statevector_gpu_stub(), &circuit, 100, 42).unwrap_err();
    assert!(matches!(
        err,
        crate::error::PrismError::BackendUnsupported { .. }
    ));
}

// Below the crossover the explicit GPU kind resolves to the host and must
// match explicit `Statevector` byte-exact through the terminal path.
#[test]
fn statevector_gpu_terminal_counts_below_crossover_match_statevector() {
    let circuit = measured_chain_circuit(6);
    let sv = run_counts_with(BackendKind::Statevector, &circuit, 200, 42)
        .unwrap()
        .0;
    let gpu = run_counts_with(statevector_gpu_stub(), &circuit, 200, 42)
        .unwrap()
        .0;
    assert_eq!(sv, gpu);
}

#[test]
fn auto_gpu_expectation_matches_auto_on_stub() {
    let circuit = chain_circuit(16);
    let auto_vals =
        run_expectation_values_with(BackendKind::Auto, &circuit, &observables(), 42).unwrap();
    let gpu_vals =
        run_expectation_values_with(auto_gpu_stub(), &circuit, &observables(), 42).unwrap();
    assert_expectations_close(&auto_vals, &gpu_vals);
}

// Explicit `StatevectorGpu` expectation values resolve hard above the
// crossover, so the stub's failed allocation surfaces.
#[test]
fn statevector_gpu_expectation_hard_above_crossover_on_stub() {
    let circuit = chain_circuit(16);
    let err = run_expectation_values_with(statevector_gpu_stub(), &circuit, &observables(), 42)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::error::PrismError::BackendUnsupported { .. }
    ));
}

#[test]
fn statevector_gpu_expectation_below_crossover_matches_statevector() {
    let circuit = chain_circuit(6);
    let sv = run_expectation_values_with(BackendKind::Statevector, &circuit, &observables(), 42)
        .unwrap();
    let gpu =
        run_expectation_values_with(statevector_gpu_stub(), &circuit, &observables(), 42).unwrap();
    assert_expectations_close(&sv, &gpu);
}

// AutoGpu + non-Pauli noise resolves the trajectory plan through the
// capability table. On the stub the VRAM gate fails closed, so the run
// must be byte-identical to `Auto`.
#[test]
fn auto_gpu_general_noise_matches_auto_on_stub() {
    let circuit = measured_chain_circuit(14);
    let noise = noise::NoiseModel::with_amplitude_damping(&circuit, 0.05);
    let auto_shots = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 16, 42).unwrap();
    let gpu_shots = run_shots_with_noise(auto_gpu_stub(), &circuit, &noise, 16, 42).unwrap();
    assert_eq!(auto_shots.shots, gpu_shots.shots);
}

// A non-entangling circuit resolves to the product-state family for
// general noise under AutoGpu, per the capability table (no GPU row).
#[test]
fn auto_gpu_general_noise_product_circuit_matches_auto() {
    let mut circuit = Circuit::new(6, 6);
    for q in 0..6 {
        circuit.add_gate(Gate::Rx(0.4), &[q]);
    }
    for q in 0..6 {
        circuit.add_measure(q, q);
    }
    let noise = noise::NoiseModel::with_amplitude_damping(&circuit, 0.05);
    let auto_shots = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 64, 42).unwrap();
    let gpu_shots = run_shots_with_noise(auto_gpu_stub(), &circuit, &noise, 64, 42).unwrap();
    assert_eq!(auto_shots.shots, gpu_shots.shots);
}
