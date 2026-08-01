use super::*;
use crate::gates::Gate;
use crate::gpu::GpuContext;

fn noisy_circuit(n: usize) -> Circuit {
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

// AutoGpu + non-Pauli noise resolves the trajectory plan through the
// capability table. On the stub the VRAM gate fails closed, so the run
// must be byte-identical to `Auto`.
#[test]
fn auto_gpu_general_noise_matches_auto_on_stub() {
    let circuit = noisy_circuit(14);
    let noise = noise::NoiseModel::with_amplitude_damping(&circuit, 0.05);
    let auto_shots = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 16, 42).unwrap();
    let gpu_shots = run_shots_with_noise(
        BackendKind::AutoGpu {
            context: GpuContext::stub_for_tests(),
        },
        &circuit,
        &noise,
        16,
        42,
    )
    .unwrap();
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
    let gpu_shots = run_shots_with_noise(
        BackendKind::AutoGpu {
            context: GpuContext::stub_for_tests(),
        },
        &circuit,
        &noise,
        64,
        42,
    )
    .unwrap();
    assert_eq!(auto_shots.shots, gpu_shots.shots);
}
