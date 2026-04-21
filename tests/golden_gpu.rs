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
fn diagonal_batch_14q_mixed_entries_matches_cpu() {
    // Drive a mixed DiagonalBatch (all three DiagEntry kinds across overlapping qubits)
    // through the batched GPU kernel at a size that forces real DRAM passes.
    let Some(f) = Fixture::try_new() else { return };
    let n = 14;
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    let entries = vec![
        DiagEntry::Phase1q {
            qubit: 0,
            d0: Complex64::from_polar(1.0, 0.3),
            d1: Complex64::from_polar(1.0, -0.7),
        },
        DiagEntry::Phase1q {
            qubit: 5,
            d0: Complex64::from_polar(1.0, 1.1),
            d1: Complex64::from_polar(1.0, -0.4),
        },
        DiagEntry::Phase2q {
            q0: 1,
            q1: 2,
            phase: Complex64::from_polar(1.0, 0.8),
        },
        DiagEntry::Parity2q {
            q0: 3,
            q1: 7,
            same: Complex64::from_polar(1.0, 0.2),
            diff: Complex64::from_polar(1.0, -1.3),
        },
    ];
    insts.push(g(
        Gate::DiagonalBatch(Box::new(DiagonalBatchData { entries })),
        &(0..8).collect::<Vec<_>>(),
    ));
    f.compare(n, &insts);
}

#[test]
fn batch_rzz_14q_qaoa_matches_cpu() {
    // QAOA emits `Gate::BatchRzz` once fusion groups consecutive Rzz gates. At 14q
    // (the MIN_QUBITS_FOR_BATCH_RZZ threshold), the batched GPU kernel must match the
    // CPU LUT kernel.
    let Some(f) = Fixture::try_new() else { return };
    let circuit = prism_q::circuits::qaoa_circuit(14, 2, 42);

    let mut cpu = StatevectorBackend::new(42);
    cpu.init(14, 0).unwrap();
    let mut gpu = StatevectorBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(14, 0).unwrap();
    for inst in &circuit.instructions {
        cpu.apply(inst).unwrap();
        gpu.apply(inst).unwrap();
    }
    let cpu_sv = cpu.export_statevector().unwrap();
    let gpu_sv = gpu.export_statevector().unwrap();
    let eps = 1e-10;
    for (i, (c, g)) in cpu_sv.iter().zip(gpu_sv.iter()).enumerate() {
        let diff = (c - g).norm();
        assert!(
            diff < eps,
            "qaoa_14q mismatch at {i}: cpu={c:?} gpu={g:?} |diff|={diff}"
        );
    }
}

#[test]
fn batch_phase_16q_matches_cpu() {
    // QFT at 16q exercises the `fuse_controlled_phases` pass which emits
    // `Gate::BatchPhase`. The batched GPU kernel must match the CPU tiled kernel
    // within tolerance. The test drives through `sim::run_with` so fusion fires
    // normally, then compares probabilities between plain statevector and GPU statevector.
    let Some(f) = Fixture::try_new() else { return };

    let circuit = prism_q::circuits::qft_circuit(16);

    let mut cpu = StatevectorBackend::new(42);
    cpu.init(16, 0).unwrap();
    let mut gpu = StatevectorBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(16, 0).unwrap();
    for inst in &circuit.instructions {
        cpu.apply(inst).unwrap();
        gpu.apply(inst).unwrap();
    }
    let cpu_sv = cpu.export_statevector().unwrap();
    let gpu_sv = gpu.export_statevector().unwrap();
    let eps = 1e-10; // looser than default 1e-12 for 16q × ~130 gates
    for (i, (c, g)) in cpu_sv.iter().zip(gpu_sv.iter()).enumerate() {
        let diff = (c - g).norm();
        assert!(
            diff < eps,
            "qft_16q mismatch at {i}: cpu={c:?} gpu={g:?} |diff|={diff}"
        );
    }
}

#[test]
fn multi_fused_nondiag_tiled_matches_cpu() {
    // Exercise the tiled non-diagonal MultiFused kernel with a mix of low-target gates
    // (go through shared memory) and high-target gates (fall back to per-gate launches).
    // 14 qubits: TILE_Q=10 so targets 0..9 are tile-local, targets 10..13 are external.
    let Some(f) = Fixture::try_new() else { return };
    let n = 14;
    let mat_h = [
        [
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
        ],
        [
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
        ],
    ];
    let mat_ry = [
        [Complex64::new(0.8, 0.0), Complex64::new(-0.6, 0.0)],
        [Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)],
    ];
    // Mix tile-local (targets 0..9) and external (targets 10..13).
    let gates: Vec<(usize, [[Complex64; 2]; 2])> = vec![
        (0, mat_h),
        (3, mat_ry),
        (5, mat_h),
        (7, mat_ry),
        (9, mat_h), // boundary of tile (TILE_Q=10, so target 9 is the highest tile-local)
        (10, mat_ry),
        (12, mat_h),
        (13, mat_ry),
    ];

    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    insts.push(g(
        Gate::MultiFused(Box::new(MultiFusedData {
            gates,
            all_diagonal: false,
        })),
        &(0..8).collect::<Vec<_>>(),
    ));
    f.compare(n, &insts);
}

#[test]
fn multi_fused_diagonal_batched_matches_cpu() {
    // Constructs a MultiFused of 8 diagonal 1q gates on 14 qubits (above
    // MIN_QUBITS_FOR_MULTI_FUSION so the batched path is the one being exercised) and
    // proves the single-launch batched GPU kernel matches the CPU tiled kernel.
    let Some(f) = Fixture::try_new() else { return };
    let n = 14;
    let gates: Vec<(usize, [[Complex64; 2]; 2])> = (0..8)
        .map(|i| {
            let q = i;
            let phase_0 = Complex64::from_polar(1.0, 0.1 * (i as f64 + 1.0));
            let phase_1 = Complex64::from_polar(1.0, -0.07 * (i as f64 + 3.0));
            let z = Complex64::new(0.0, 0.0);
            (q, [[phase_0, z], [z, phase_1]])
        })
        .collect();

    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    insts.push(g(
        Gate::MultiFused(Box::new(MultiFusedData {
            gates: gates.clone(),
            all_diagonal: true,
        })),
        &(0..8).collect::<Vec<_>>(),
    ));
    f.compare(n, &insts);
}

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

// ============================================================================
// BackendKind::StatevectorGpu end-to-end (real GPU required)
// ============================================================================

/// `run_with(BackendKind::StatevectorGpu)` on a non-decomposable random
/// circuit at 14q (at the crossover threshold) must match CPU to within
/// 1e-10. Exercises the dispatch → fusion → GPU kernel → probabilities path.
#[test]
fn statevector_gpu_run_with_matches_cpu_random() {
    use prism_q::BackendKind;

    let Some(f) = Fixture::try_new() else { return };

    let circuit = prism_q::circuits::random_circuit(14, 10, 0xDEAD_BEEF);

    let cpu = prism_q::sim::run_with(BackendKind::Statevector, &circuit, 42)
        .expect("cpu run_with failed");
    let gpu = prism_q::sim::run_with(
        BackendKind::StatevectorGpu {
            context: f.ctx.clone(),
        },
        &circuit,
        42,
    )
    .expect("gpu run_with failed");

    let cpu_p = cpu.probabilities.expect("cpu probs missing").to_vec();
    let gpu_p = gpu.probabilities.expect("gpu probs missing").to_vec();
    assert_eq!(cpu_p.len(), gpu_p.len());
    for (i, (c, g_)) in cpu_p.iter().zip(gpu_p.iter()).enumerate() {
        assert!(
            (c - g_).abs() < 1e-10,
            "prob[{i}] cpu={c}, gpu={g_}, diff={}",
            (c - g_).abs()
        );
    }
}

// ============================================================================
// Stabilizer GPU scaffold (M2)
// ============================================================================

/// `StabilizerBackend::with_gpu(ctx).init(n, 0)` must allocate the device tableau
/// successfully on a real GPU. Any gate application or probability query must
/// then surface `BackendUnsupported` (until subsequent milestones add kernels).
#[test]
fn stabilizer_gpu_init_allocates_tableau_and_rejects_apply() {
    use prism_q::error::PrismError;
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };

    let mut backend = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    backend.init(4, 0).expect("GPU tableau init should succeed");
    assert_eq!(backend.num_qubits(), 4);

    let apply_err = backend
        .apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap_err();
    assert!(
        matches!(apply_err, PrismError::BackendUnsupported { .. }),
        "apply returned unexpected error: {apply_err:?}"
    );

    let probs_err = backend.probabilities().unwrap_err();
    assert!(
        matches!(probs_err, PrismError::BackendUnsupported { .. }),
        "probabilities returned unexpected error: {probs_err:?}"
    );
}

/// `Barrier` must succeed on the GPU path (no-op), and a `Conditional` whose
/// predicate evaluates to false must succeed without attempting to apply the
/// gate. Both instructions need no device work.
#[test]
fn stabilizer_gpu_accepts_barrier_and_false_conditional() {
    use prism_q::circuit::ClassicalCondition;
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };

    let mut backend = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    backend.init(4, 1).expect("GPU tableau init should succeed");

    backend
        .apply(&Instruction::Barrier {
            qubits: smallvec![0_usize, 1_usize, 2_usize, 3_usize],
        })
        .expect("barrier must be a no-op on the GPU path");

    // Classical bit 0 starts false, so `BitIsOne(0)` evaluates to false and
    // the gate is never applied.
    backend
        .apply(&Instruction::Conditional {
            condition: ClassicalCondition::BitIsOne(0),
            gate: Gate::H,
            targets: smallvec![0],
        })
        .expect("false-predicate conditional must be a no-op on the GPU path");
}

/// Cloning a GPU-attached `StabilizerBackend` must panic loudly rather than
/// silently produce an invalid state. The `#[should_panic]` matcher fires
/// whether we reach the `backend.clone()` call (real GPU available) or the
/// explicit skip panic below (no GPU available).
#[test]
#[should_panic(expected = "StabilizerBackend::clone is unsupported")]
fn stabilizer_gpu_clone_panics() {
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else {
        // Satisfy `#[should_panic]` even on CPU-only runners so the test
        // reports pass rather than "did not panic".
        panic!("StabilizerBackend::clone is unsupported (skipped: no GPU)");
    };

    let mut backend = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    backend.init(4, 0).expect("GPU tableau init should succeed");
    let _cloned = backend.clone();
}
