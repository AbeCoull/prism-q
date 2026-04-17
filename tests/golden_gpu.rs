//! Per-kernel GPU vs CPU amplitude equivalence tests.
//!
//! Runs each gate-dispatch path on both backends starting from |0…0⟩ (or a deliberately
//! chosen non-trivial initial state) and compares the resulting amplitudes within 1e-12.
//!
//! Skips with a printed message when no usable GPU is available. This keeps the suite green
//! on CPU-only machines and catches real kernel regressions on CUDA-capable runners.

#![cfg(feature = "gpu")]

use num_complex::Complex64;

use prism_q::backend::Backend;
use prism_q::circuit::{smallvec, Instruction, SmallVec};
use prism_q::gates::{DiagEntry, DiagonalBatchData, Gate, McuData, MultiFusedData};
use prism_q::gpu::GpuContext;
use prism_q::StatevectorBackend;

const EPS: f64 = 1e-12;

struct Fixture {
    ctx: std::sync::Arc<GpuContext>,
}

impl Fixture {
    fn try_new() -> Option<Self> {
        match GpuContext::new(0) {
            Ok(ctx) => Some(Self { ctx }),
            Err(e) => {
                eprintln!("SKIP: no usable GPU ({e})");
                None
            }
        }
    }

    fn compare(&self, num_qubits: usize, instructions: &[Instruction]) {
        let mut cpu = StatevectorBackend::new(42);
        cpu.init(num_qubits, 0).unwrap();
        let mut gpu = StatevectorBackend::new(42).with_gpu(self.ctx.clone());
        gpu.init(num_qubits, 0).unwrap();
        for inst in instructions {
            cpu.apply(inst).unwrap();
            gpu.apply(inst).unwrap();
        }
        let cpu_sv = cpu.export_statevector().unwrap();
        let gpu_sv = gpu.export_statevector().unwrap();
        assert_eq!(cpu_sv.len(), gpu_sv.len());
        for (i, (c, g)) in cpu_sv.iter().zip(gpu_sv.iter()).enumerate() {
            let diff = (c - g).norm();
            assert!(
                diff < EPS,
                "amplitude mismatch at index {i}: cpu={c:?}, gpu={g:?}, |diff|={diff}"
            );
        }
    }
}

fn g(gate: Gate, targets: &[usize]) -> Instruction {
    let mut tv: SmallVec<[usize; 4]> = smallvec![];
    tv.extend_from_slice(targets);
    Instruction::Gate { gate, targets: tv }
}

// ============================================================================
// Single-qubit kernels
// ============================================================================

#[test]
fn single_qubit_gates_all_targets() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    // Seed with a non-trivial superposition so each gate has something to act on.
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    // Exercise every 1q variant on each target.
    let targets = [0usize, 1, 2, 3];
    let gates_1q = [
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::SX,
        Gate::SXdg,
        Gate::Rx(0.37),
        Gate::Ry(1.11),
        Gate::Rz(-0.62),
        Gate::P(0.44),
    ];
    for t in targets {
        for gate in &gates_1q {
            insts.push(g(gate.clone(), &[t]));
        }
    }
    f.compare(n, &insts);
}

#[test]
fn fused_1q_matrix_arbitrary() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    // Non-unitary coefficients are fine here; we only compare CPU==GPU, not a unitary property.
    let mat = [
        [Complex64::new(0.3, 0.1), Complex64::new(-0.2, 0.4)],
        [Complex64::new(0.5, -0.3), Complex64::new(0.1, 0.2)],
    ];
    let mut insts = vec![g(Gate::H, &[0]), g(Gate::H, &[1])];
    for t in 0..n {
        insts.push(g(Gate::Fused(Box::new(mat)), &[t]));
    }
    f.compare(n, &insts);
}

// ============================================================================
// Two-qubit kernels
// ============================================================================

#[test]
fn cx_cz_swap_various_pairs() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    // Pairs covering q0<q1 and q0>q1.
    let pairs: &[(usize, usize)] = &[(0, 1), (1, 2), (0, 3), (3, 0), (2, 1)];
    for &(a, b) in pairs {
        insts.push(g(Gate::Cx, &[a, b]));
        insts.push(g(Gate::Cz, &[a, b]));
        insts.push(g(Gate::Swap, &[a, b]));
    }
    f.compare(n, &insts);
}

#[test]
fn rzz_various_pairs() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    let mut insts = vec![g(Gate::H, &[0]), g(Gate::H, &[1]), g(Gate::H, &[2])];
    for (a, b, theta) in [(0, 1, 0.3), (2, 0, -1.1), (3, 2, 1.7), (1, 3, 0.9)] {
        insts.push(g(Gate::Rzz(theta), &[a, b]));
    }
    f.compare(n, &insts);
}

// ============================================================================
// Controlled-unitary kernels
// ============================================================================

#[test]
fn cu_arbitrary_matrix() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    let mat = [
        [Complex64::new(0.6, -0.1), Complex64::new(-0.3, 0.2)],
        [Complex64::new(0.2, 0.4), Complex64::new(0.7, -0.2)],
    ];
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    insts.push(g(Gate::Cu(Box::new(mat)), &[0, 2]));
    insts.push(g(Gate::Cu(Box::new(mat)), &[3, 1]));
    f.compare(n, &insts);
}

#[test]
fn cu_controlled_phase_shortcut() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    // A diagonal CU is treated as a controlled-phase (cu_phase kernel).
    let phase = Complex64::from_polar(1.0, 0.73);
    let diag = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), phase],
    ];
    let mut insts = vec![g(Gate::H, &[0]), g(Gate::H, &[2]), g(Gate::X, &[3])];
    insts.push(g(Gate::Cu(Box::new(diag)), &[0, 2]));
    insts.push(g(Gate::Cu(Box::new(diag)), &[3, 1]));
    f.compare(n, &insts);
}

#[test]
fn mcu_toffoli_and_generic() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 5;
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    // Toffoli (2 controls, X target) via Mcu with an X matrix.
    let x_mat = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let mcu_toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    insts.push(g(mcu_toffoli, &[0, 1, 2]));

    // Three-control arbitrary unitary.
    let mat = [
        [Complex64::new(0.4, 0.3), Complex64::new(-0.2, 0.5)],
        [Complex64::new(0.5, -0.2), Complex64::new(0.3, 0.4)],
    ];
    let mcu_3 = Gate::Mcu(Box::new(McuData {
        mat,
        num_controls: 3,
    }));
    insts.push(g(mcu_3, &[0, 1, 3, 4]));
    f.compare(n, &insts);
}

#[test]
fn mcu_phase_shortcut() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 5;
    let phase = Complex64::from_polar(1.0, -1.2);
    let diag = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), phase],
    ];
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    let mcu = Gate::Mcu(Box::new(McuData {
        mat: diag,
        num_controls: 3,
    }));
    insts.push(g(mcu, &[0, 1, 2, 4]));
    f.compare(n, &insts);
}

// ============================================================================
// Fused / batched gates
// ============================================================================

#[test]
fn fused_2q_asymmetric_matrix() {
    // Critical: non-diagonal 4x4 with all 16 entries distinct detects any basis-ordering bug
    // between CPU (PreparedGate2q) and GPU apply_fused_2q.
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    let mut mat = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (r, row) in mat.iter_mut().enumerate() {
        for (c, entry) in row.iter_mut().enumerate() {
            *entry = Complex64::new(0.1 * (r as f64 + 1.0), 0.07 * (c as f64 + 1.0));
        }
    }
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    insts.push(g(Gate::Fused2q(Box::new(mat)), &[0, 2]));
    insts.push(g(Gate::Fused2q(Box::new(mat)), &[3, 1]));
    f.compare(n, &insts);
}

#[test]
fn multi_fused_mixed_gates() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    let mat_a = [
        [Complex64::new(0.3, 0.1), Complex64::new(-0.2, 0.4)],
        [Complex64::new(0.5, -0.3), Complex64::new(0.1, 0.2)],
    ];
    let mat_b = [
        [Complex64::new(0.9, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ];
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    insts.push(g(
        Gate::MultiFused(Box::new(MultiFusedData {
            gates: vec![(0, mat_a), (1, mat_b), (2, mat_a), (3, mat_b)],
            all_diagonal: false,
        })),
        &[0, 1, 2, 3],
    ));
    f.compare(n, &insts);
}

#[test]
fn diagonal_batch_all_entry_kinds() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    let entries = vec![
        DiagEntry::Phase1q {
            qubit: 0,
            d0: Complex64::from_polar(1.0, 0.2),
            d1: Complex64::from_polar(1.0, -0.3),
        },
        DiagEntry::Phase2q {
            q0: 1,
            q1: 2,
            phase: Complex64::from_polar(1.0, 0.5),
        },
        DiagEntry::Parity2q {
            q0: 0,
            q1: 3,
            same: Complex64::from_polar(1.0, 0.1),
            diff: Complex64::from_polar(1.0, -0.4),
        },
    ];
    insts.push(g(
        Gate::DiagonalBatch(Box::new(DiagonalBatchData { entries })),
        &[0, 1, 2, 3],
    ));
    f.compare(n, &insts);
}

// ============================================================================
// Measurement + reset
// ============================================================================

#[test]
fn measurement_deterministic_with_fixed_seed() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 3;
    let mut cpu = StatevectorBackend::new(42);
    cpu.init(n, n).unwrap();
    let mut gpu = StatevectorBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(n, n).unwrap();

    let prep = [g(Gate::H, &[0]), g(Gate::Cx, &[0, 1]), g(Gate::H, &[2])];
    for inst in &prep {
        cpu.apply(inst).unwrap();
        gpu.apply(inst).unwrap();
    }
    for q in 0..n {
        cpu.apply(&Instruction::Measure {
            qubit: q,
            classical_bit: q,
        })
        .unwrap();
        gpu.apply(&Instruction::Measure {
            qubit: q,
            classical_bit: q,
        })
        .unwrap();
    }
    assert_eq!(cpu.classical_results(), gpu.classical_results());
    // Export statevectors and verify residual state matches (post-collapse).
    let cpu_sv = cpu.export_statevector().unwrap();
    let gpu_sv = gpu.export_statevector().unwrap();
    for (i, (c, g_)) in cpu_sv.iter().zip(gpu_sv.iter()).enumerate() {
        let diff = (c - g_).norm();
        assert!(
            diff < EPS,
            "post-measurement amplitude mismatch at {i}: {c:?} vs {g_:?}"
        );
    }
}

#[test]
fn reset_to_zero() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 3;
    let mut cpu = StatevectorBackend::new(42);
    cpu.init(n, 0).unwrap();
    let mut gpu = StatevectorBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(n, 0).unwrap();

    // Put the register into |1⟩|+⟩|1⟩, then reset q0 and q2.
    let prep = [g(Gate::X, &[0]), g(Gate::H, &[1]), g(Gate::X, &[2])];
    for inst in &prep {
        cpu.apply(inst).unwrap();
        gpu.apply(inst).unwrap();
    }
    cpu.apply(&Instruction::Reset { qubit: 0 }).unwrap();
    gpu.apply(&Instruction::Reset { qubit: 0 }).unwrap();
    cpu.apply(&Instruction::Reset { qubit: 2 }).unwrap();
    gpu.apply(&Instruction::Reset { qubit: 2 }).unwrap();

    let cpu_sv = cpu.export_statevector().unwrap();
    let gpu_sv = gpu.export_statevector().unwrap();
    for (i, (c, g_)) in cpu_sv.iter().zip(gpu_sv.iter()).enumerate() {
        let diff = (c - g_).norm();
        assert!(
            diff < EPS,
            "reset amplitude mismatch at {i}: {c:?} vs {g_:?}"
        );
    }
}

// ============================================================================
// Probabilities readback
// ============================================================================

#[test]
fn probabilities_match_cpu_on_random_circuit() {
    let Some(f) = Fixture::try_new() else { return };
    let circuit = prism_q::circuits::random_circuit(6, 30, 42);
    let mut cpu = StatevectorBackend::new(42);
    cpu.init(circuit.num_qubits, 0).unwrap();
    let mut gpu = StatevectorBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(circuit.num_qubits, 0).unwrap();
    for inst in &circuit.instructions {
        cpu.apply(inst).unwrap();
        gpu.apply(inst).unwrap();
    }
    let cpu_p = cpu.probabilities().unwrap();
    let gpu_p = gpu.probabilities().unwrap();
    assert_eq!(cpu_p.len(), gpu_p.len());
    for (i, (c, g_)) in cpu_p.iter().zip(gpu_p.iter()).enumerate() {
        assert!(
            (c - g_).abs() < EPS,
            "prob[{i}] mismatch: cpu={c}, gpu={g_}"
        );
    }
}
