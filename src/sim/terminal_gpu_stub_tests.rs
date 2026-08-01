use super::*;
use crate::gates::Gate;
use crate::gpu::GpuContext;

fn terminal_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::Rx(0.3), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
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

// The terminal fast path resolves the accel through the capability table.
// On the stub the VRAM gate fails closed, so `AutoGpu` must take the
// identical host path as `Auto`: same sampler, same RNG stream, byte-equal
// counts.
#[test]
fn auto_gpu_terminal_counts_match_auto_on_stub() {
    let circuit = terminal_circuit(16);
    let auto_counts = run_counts_with(BackendKind::Auto, &circuit, 500, 42).unwrap();
    let gpu_counts = run_counts_with(auto_gpu_stub(), &circuit, 500, 42).unwrap();
    assert_eq!(auto_counts, gpu_counts);
}

#[test]
fn auto_gpu_terminal_shots_match_auto_on_stub() {
    let circuit = terminal_circuit(16);
    let auto_shots = run_shots_with(BackendKind::Auto, &circuit, 64, 42).unwrap();
    let gpu_shots = run_shots_with(auto_gpu_stub(), &circuit, 64, 42).unwrap();
    assert_eq!(auto_shots.shots, gpu_shots.shots);
}

// Explicit `StatevectorGpu` is a terminal-fast-path candidate. Above the
// crossover it resolves hard, so the stub's failed allocation surfaces
// instead of falling back, proving the device path was reached.
#[test]
fn statevector_gpu_terminal_counts_hard_above_crossover_on_stub() {
    let circuit = terminal_circuit(16);
    let kind = BackendKind::StatevectorGpu {
        context: GpuContext::stub_for_tests(),
    };
    let err = run_counts_with(kind, &circuit, 100, 42).unwrap_err();
    assert!(matches!(
        err,
        crate::error::PrismError::BackendUnsupported { .. }
    ));
}

// Below the crossover the explicit GPU kind resolves to the host and must
// match explicit `Statevector` byte-exact through the terminal path.
#[test]
fn statevector_gpu_terminal_counts_below_crossover_match_statevector() {
    let circuit = terminal_circuit(6);
    let kind = BackendKind::StatevectorGpu {
        context: GpuContext::stub_for_tests(),
    };
    let sv = run_counts_with(BackendKind::Statevector, &circuit, 200, 42).unwrap();
    let gpu = run_counts_with(kind, &circuit, 200, 42).unwrap();
    assert_eq!(sv, gpu);
}
