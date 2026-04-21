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
// Stabilizer GPU scaffold
// ============================================================================

/// `StabilizerBackend::with_gpu(ctx).init(n, 0)` must allocate the device tableau
/// successfully on a real GPU, and applying a single Clifford gate must succeed
/// via the GPU kernel path. The full `apply` / `probabilities` / `measure` /
/// `reset` surface is wired through the GPU route.
#[test]
fn stabilizer_gpu_init_allocates_tableau_and_accepts_gate() {
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };

    let mut backend = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    backend.init(4, 0).expect("GPU tableau init should succeed");
    assert_eq!(backend.num_qubits(), 4);

    backend
        .apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .expect("H should dispatch to the GPU kernel");

    let probs = backend
        .probabilities()
        .expect("probabilities should work via host copy-back");
    assert_eq!(probs.len(), 16);
    // H on q0 from |0000⟩ gives 50/50 between basis 0 and basis 1.
    assert!((probs[0] - 0.5).abs() < 1e-12);
    assert!((probs[1] - 0.5).abs() < 1e-12);
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

// ============================================================================
// Stabilizer GPU Clifford gate kernels
// ============================================================================

/// Apply the given instructions to fresh CPU and GPU `StabilizerBackend`s and
/// assert the resulting tableau state (xz, phase) matches byte for byte.
fn stab_compare(f: &Fixture, num_qubits: usize, instructions: &[Instruction]) {
    use prism_q::StabilizerBackend;

    let mut cpu = StabilizerBackend::new(42);
    cpu.init(num_qubits, 0).unwrap();
    for inst in instructions {
        cpu.apply(inst).unwrap();
    }
    let (cpu_xz, cpu_phase) = cpu.export_tableau().unwrap();

    let mut gpu = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(num_qubits, 0).unwrap();
    for inst in instructions {
        gpu.apply(inst).unwrap();
    }
    let (gpu_xz, gpu_phase) = gpu.export_tableau().unwrap();

    assert_eq!(
        cpu_xz,
        gpu_xz,
        "xz mismatch after {} instructions",
        instructions.len()
    );
    assert_eq!(
        cpu_phase,
        gpu_phase,
        "phase mismatch after {} instructions",
        instructions.len()
    );
}

#[test]
fn stabilizer_gpu_single_qubit_gates_all_targets() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    // Seed the tableau with an H on every qubit so subsequent gates have
    // non-trivial (x, z) bit patterns to act on.
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    let gates_1q = [
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::SX,
        Gate::SXdg,
    ];
    // Apply each 1q variant to each target qubit.
    for t in 0..n {
        for gate in &gates_1q {
            insts.push(g(gate.clone(), &[t]));
        }
    }
    stab_compare(&f, n, &insts);
}

/// `Gate::Id` is dispatched as a no-op: no kernel launch, tableau unchanged.
/// Driving a mix of Id and real gates must produce the same tableau as a
/// run with Id omitted.
#[test]
fn stabilizer_gpu_id_is_noop() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 3;
    let real = vec![g(Gate::H, &[0]), g(Gate::Cx, &[0, 1]), g(Gate::S, &[2])];
    let with_id = vec![
        g(Gate::Id, &[0]),
        g(Gate::H, &[0]),
        g(Gate::Id, &[1]),
        g(Gate::Cx, &[0, 1]),
        g(Gate::Id, &[2]),
        g(Gate::S, &[2]),
        g(Gate::Id, &[0]),
    ];
    // Both programs must leave the GPU tableau in the same state.
    use prism_q::StabilizerBackend;
    let mut a = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    a.init(n, 0).unwrap();
    for inst in &real {
        a.apply(inst).unwrap();
    }
    let (a_xz, a_phase) = a.export_tableau().unwrap();

    let mut b = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    b.init(n, 0).unwrap();
    for inst in &with_id {
        b.apply(inst).unwrap();
    }
    let (b_xz, b_phase) = b.export_tableau().unwrap();

    assert_eq!(a_xz, b_xz, "Id must not change xz");
    assert_eq!(a_phase, b_phase, "Id must not change phase");
}

#[test]
fn stabilizer_gpu_two_qubit_gates_various_pairs() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 4;
    let mut insts = vec![];
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    // Cover q0<q1 and q0>q1, adjacent and non-adjacent pairs.
    let pairs: &[(usize, usize)] = &[(0, 1), (1, 2), (0, 3), (3, 0), (2, 1)];
    for &(a, b) in pairs {
        insts.push(g(Gate::Cx, &[a, b]));
        insts.push(g(Gate::Cz, &[a, b]));
        insts.push(g(Gate::Swap, &[a, b]));
    }
    stab_compare(&f, n, &insts);
}

#[test]
fn stabilizer_gpu_cross_word_targets() {
    // Exercises `word_idx != 0` paths by operating on qubits >= 64.
    let Some(f) = Fixture::try_new() else { return };
    let n = 80;
    // Seed a Bell pair straddling the 64-qubit word boundary, then exercise
    // every gate variant on qubits in both word halves.
    let insts = vec![
        g(Gate::H, &[63]),
        g(Gate::H, &[64]),
        g(Gate::Cx, &[63, 64]),
        g(Gate::Cz, &[65, 70]),
        g(Gate::Swap, &[72, 5]),
        g(Gate::S, &[79]),
        g(Gate::Sdg, &[64]),
        g(Gate::Z, &[63]),
        g(Gate::X, &[78]),
        g(Gate::Y, &[70]),
        g(Gate::SX, &[65]),
        g(Gate::SXdg, &[66]),
    ];
    stab_compare(&f, n, &insts);
}

#[test]
fn stabilizer_gpu_batch_1000_gates_matches_cpu() {
    // Runs ~1000 mixed Clifford gates via `apply_instructions` so the GPU
    // path flushes the queued op list through `stab_apply_batch` in a single
    // launch, then compares the post-batch device tableau against CPU
    // byte for byte. Every opcode consumed by the batch kernel is exercised
    // multiple times, including targets on both word halves.
    use prism_q::StabilizerBackend;
    let Some(f) = Fixture::try_new() else { return };
    let n = 80;

    let ones = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];
    let twos = [Gate::Cx, Gate::Cz, Gate::Swap];
    let mut insts = Vec::with_capacity(1024);
    for q in 0..n {
        insts.push(g(Gate::H, &[q]));
    }
    // Deterministic mix: cycle through opcodes and targets so every (op,
    // word-half) combination shows up many times within the batch.
    for step in 0..1000 {
        if step % 4 == 0 {
            let a = step % n;
            let b = (step * 7 + 13) % n;
            let g2 = twos[(step / 4) % twos.len()].clone();
            if a != b {
                insts.push(g(g2, &[a, b]));
                continue;
            }
        }
        let g1 = ones[step % ones.len()].clone();
        let t = (step * 5 + 1) % n;
        insts.push(g(g1, &[t]));
    }
    // Also drop in SX, SXdg, and cross-word CX/CZ/SWAP so the rarer opcodes
    // are covered within the same batch.
    insts.push(g(Gate::SX, &[77]));
    insts.push(g(Gate::SXdg, &[2]));
    insts.push(g(Gate::Cx, &[63, 64]));
    insts.push(g(Gate::Cz, &[5, 70]));
    insts.push(g(Gate::Swap, &[10, 72]));

    let mut cpu = StabilizerBackend::new(42);
    cpu.init(n, 0).unwrap();
    cpu.apply_instructions(&insts).unwrap();
    let (cpu_xz, cpu_phase) = cpu.export_tableau().unwrap();

    let mut gpu = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(n, 0).unwrap();
    gpu.apply_instructions(&insts).unwrap();
    let (gpu_xz, gpu_phase) = gpu.export_tableau().unwrap();

    assert_eq!(cpu_xz, gpu_xz, "batched xz mismatch");
    assert_eq!(cpu_phase, gpu_phase, "batched phase mismatch");
}

#[test]
fn stabilizer_gpu_bell_and_ghz() {
    let Some(f) = Fixture::try_new() else { return };
    // Bell state on 2 qubits.
    let bell = [g(Gate::H, &[0]), g(Gate::Cx, &[0, 1])];
    stab_compare(&f, 2, &bell);

    // 4q GHZ state.
    let mut ghz = vec![g(Gate::H, &[0])];
    for i in 0..3 {
        ghz.push(g(Gate::Cx, &[i, i + 1]));
    }
    stab_compare(&f, 4, &ghz);
}

#[test]
fn stabilizer_gpu_rejects_non_clifford_via_gpu_path() {
    use prism_q::error::PrismError;
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };
    let mut backend = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    backend.init(2, 0).unwrap();

    // T is non-Clifford; must return BackendUnsupported with the same error
    // shape the CPU path produces.
    let err = backend.apply(&g(Gate::T, &[0])).unwrap_err();
    assert!(
        matches!(err, PrismError::BackendUnsupported { .. }),
        "non-Clifford T must be rejected; got: {err:?}"
    );
}

// ============================================================================
// Stabilizer GPU rowmul_words
// ============================================================================

/// Drive the test-only `rowmul_rows_for_testing` shim on both a CPU-mode and
/// a GPU-mode `StabilizerBackend`. After applying the same preparation circuit
/// to each, the rowmul must produce identical tableau state.
fn stab_rowmul_compare(
    f: &Fixture,
    num_qubits: usize,
    prep: &[Instruction],
    pairs: &[(usize, usize)],
) {
    use prism_q::StabilizerBackend;

    let mut cpu = StabilizerBackend::new(42);
    cpu.init(num_qubits, 0).unwrap();
    for inst in prep {
        cpu.apply(inst).unwrap();
    }
    for &(src, dst) in pairs {
        cpu.rowmul_rows_for_testing(src, dst).unwrap();
    }
    let (cpu_xz, cpu_phase) = cpu.export_tableau().unwrap();

    let mut gpu = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(num_qubits, 0).unwrap();
    for inst in prep {
        gpu.apply(inst).unwrap();
    }
    for &(src, dst) in pairs {
        gpu.rowmul_rows_for_testing(src, dst).unwrap();
    }
    let (gpu_xz, gpu_phase) = gpu.export_tableau().unwrap();

    assert_eq!(cpu_xz, gpu_xz, "rowmul xz mismatch");
    assert_eq!(cpu_phase, gpu_phase, "rowmul phase mismatch");
}

#[test]
fn stabilizer_gpu_rowmul_identity_tableau() {
    // Rowmul on a freshly initialised tableau: destabilizer row 0 (X_0) into
    // stabilizer row 2n (scratch or another row). Destabilizer and stabilizer
    // generators at the same qubit anticommute, so the g-function fires.
    let Some(f) = Fixture::try_new() else { return };
    stab_rowmul_compare(&f, 4, &[], &[(0, 4), (1, 5), (2, 6)]);
}

#[test]
fn stabilizer_gpu_rowmul_all_pauli_cases() {
    // Cover all four Pauli product cases in the g-function:
    //   I * P = P  (nonzero=0, pos=0)
    //   X * X = I, Y * Y = I, Z * Z = I  (nonzero high, pos=0)
    //   X * Y = iZ  (pos bit 1)
    //   Y * Z = iX  (pos bit 1)
    //   Z * X = iY  (pos bit 1)
    // The preparation below drives rows into mixed Pauli strings by running H,
    // S, CX on varied qubits, then the rowmul pairs cover many src/dst
    // combinations so the reduction is exercised across all four cases.
    let Some(f) = Fixture::try_new() else { return };
    let prep = vec![
        g(Gate::H, &[0]),
        g(Gate::S, &[1]),
        g(Gate::H, &[2]),
        g(Gate::Cx, &[0, 1]),
        g(Gate::Cx, &[1, 2]),
        g(Gate::S, &[2]),
        g(Gate::H, &[3]),
        g(Gate::Cz, &[0, 3]),
    ];
    let pairs = [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (0, 7), (3, 6)];
    stab_rowmul_compare(&f, 4, &prep, &pairs);
}

#[test]
fn stabilizer_gpu_rowmul_multiple_rounds() {
    // Exercise chained rowmuls: the measurement cascade rowmul
    // repeatedly into the same destination row, so correctness after multiple
    // updates matters.
    let Some(f) = Fixture::try_new() else { return };
    let prep = vec![
        g(Gate::H, &[0]),
        g(Gate::H, &[1]),
        g(Gate::H, &[2]),
        g(Gate::Cx, &[0, 1]),
        g(Gate::Cx, &[1, 2]),
        g(Gate::S, &[0]),
    ];
    // Fold rows 0, 1, 2, 3 into row 4, then row 4 into row 5, etc. Forces the
    // kernel's in-place update to be read-after-write consistent.
    let pairs = [(0, 4), (1, 4), (2, 4), (3, 4), (4, 5), (5, 6)];
    stab_rowmul_compare(&f, 4, &prep, &pairs);
}

#[test]
fn stabilizer_gpu_rowmul_cross_word() {
    // num_words > 1 exercises the word-stride loop inside the kernel. 80
    // qubits → num_words = 2.
    let Some(f) = Fixture::try_new() else { return };
    let prep = vec![
        g(Gate::H, &[0]),
        g(Gate::H, &[63]),
        g(Gate::H, &[64]),
        g(Gate::Cx, &[0, 64]),
        g(Gate::Cx, &[63, 64]),
        g(Gate::S, &[64]),
        g(Gate::S, &[0]),
    ];
    // Row pair (0, 80) straddles X-half and Z-half of both rows and covers
    // both words of a 80q tableau.
    let pairs = [(0, 80), (80, 1), (63, 143), (0, 160)];
    stab_rowmul_compare(&f, 80, &prep, &pairs);
}

#[test]
fn stabilizer_gpu_rowmul_large_tableau() {
    // 500 qubits → num_words = 8. Preparation is intentionally sparse so the
    // test runtime stays bounded; rowmul correctness at scale is the target.
    let Some(f) = Fixture::try_new() else { return };
    let mut prep = vec![];
    for q in (0..500).step_by(7) {
        prep.push(g(Gate::H, &[q]));
    }
    for q in (1..499).step_by(13) {
        prep.push(g(Gate::Cx, &[q, q + 1]));
    }
    // Rowmul pairs across the destab/stab boundary.
    let pairs = [(3, 503), (50, 550), (100, 900), (250, 0)];
    stab_rowmul_compare(&f, 500, &prep, &pairs);
}

// ============================================================================
// Stabilizer GPU measurement + probabilities
// ============================================================================

fn build_gpu_stab(
    f: &Fixture,
    num_qubits: usize,
    num_classical: usize,
) -> prism_q::StabilizerBackend {
    use prism_q::StabilizerBackend;
    let mut b = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    b.init(num_qubits, num_classical).unwrap();
    b
}

#[test]
fn stabilizer_gpu_probabilities_match_cpu() {
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };
    let circuit = &[
        g(Gate::H, &[0]),
        g(Gate::Cx, &[0, 1]),
        g(Gate::Cx, &[1, 2]),
        g(Gate::H, &[3]),
        g(Gate::Cz, &[2, 3]),
    ];

    let mut cpu = StabilizerBackend::new(42);
    cpu.init(4, 0).unwrap();
    for inst in circuit {
        cpu.apply(inst).unwrap();
    }
    let cpu_probs = cpu.probabilities().unwrap();

    let mut gpu = build_gpu_stab(&f, 4, 0);
    for inst in circuit {
        gpu.apply(inst).unwrap();
    }
    let gpu_probs = gpu.probabilities().unwrap();

    assert_eq!(cpu_probs.len(), gpu_probs.len());
    for (i, (c, g_)) in cpu_probs.iter().zip(gpu_probs.iter()).enumerate() {
        assert!((c - g_).abs() < 1e-12, "prob[{i}] cpu={c}, gpu={g_}");
    }
}

#[test]
fn stabilizer_gpu_bell_measurement_correlated() {
    let Some(f) = Fixture::try_new() else { return };
    // |Φ+⟩ = (|00⟩ + |11⟩) / √2; both bits measure equal for every seed.
    for seed in [1u64, 42, 12345, 98765] {
        use prism_q::StabilizerBackend;
        let mut b = StabilizerBackend::new(seed).with_gpu(f.ctx.clone());
        b.init(2, 2).unwrap();
        b.apply(&g(Gate::H, &[0])).unwrap();
        b.apply(&g(Gate::Cx, &[0, 1])).unwrap();
        b.apply(&Instruction::Measure {
            qubit: 0,
            classical_bit: 0,
        })
        .unwrap();
        b.apply(&Instruction::Measure {
            qubit: 1,
            classical_bit: 1,
        })
        .unwrap();
        let bits = b.classical_results();
        assert_eq!(bits[0], bits[1], "Bell pair bits must match (seed {seed})");
    }
}

#[test]
fn stabilizer_gpu_ghz_4_correlated() {
    let Some(f) = Fixture::try_new() else { return };
    // 4-qubit GHZ: all four measurement bits must be equal.
    use prism_q::StabilizerBackend;
    for seed in [1u64, 7, 31, 777] {
        let mut b = StabilizerBackend::new(seed).with_gpu(f.ctx.clone());
        b.init(4, 4).unwrap();
        b.apply(&g(Gate::H, &[0])).unwrap();
        for i in 0..3 {
            b.apply(&g(Gate::Cx, &[i, i + 1])).unwrap();
        }
        for i in 0..4 {
            b.apply(&Instruction::Measure {
                qubit: i,
                classical_bit: i,
            })
            .unwrap();
        }
        let bits = b.classical_results();
        assert!(
            bits.iter().all(|&x| x == bits[0]),
            "GHZ-4 bits must all be equal (seed {seed}), got {bits:?}"
        );
    }
}

#[test]
fn stabilizer_gpu_large_ghz_measure_all_matches_cpu() {
    // 500q GHZ, measure every qubit. CPU uses the same seed so RNG draws line
    // up per measurement. Exercises the on-device pivot / cascade / fixup
    // path plus the serial deterministic-branch kernel on a tableau too
    // large for the small-scale tests above to catch scaling regressions.
    use prism_q::StabilizerBackend;
    let Some(f) = Fixture::try_new() else { return };
    let n = 500;
    let mut circuit: Vec<Instruction> = vec![g(Gate::H, &[0])];
    for i in 0..n - 1 {
        circuit.push(g(Gate::Cx, &[i, i + 1]));
    }
    for i in 0..n {
        circuit.push(Instruction::Measure {
            qubit: i,
            classical_bit: i,
        });
    }

    let mut cpu = StabilizerBackend::new(42);
    cpu.init(n, n).unwrap();
    for inst in &circuit {
        cpu.apply(inst).unwrap();
    }

    let mut gpu = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(n, n).unwrap();
    for inst in &circuit {
        gpu.apply(inst).unwrap();
    }

    assert_eq!(
        cpu.classical_results(),
        gpu.classical_results(),
        "GHZ-{n} measure-all bits must match byte for byte"
    );
    // GHZ sanity: every measurement bit equals the first.
    let bits = gpu.classical_results();
    assert!(
        bits.iter().all(|&x| x == bits[0]),
        "GHZ-{n} bits must all equal bits[0]"
    );
}

#[test]
fn stabilizer_gpu_reset_collapses_to_zero() {
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };
    let mut b = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    b.init(2, 2).unwrap();
    // Prepare |1⟩⊗|1⟩, then reset q0 and measure both.
    b.apply(&g(Gate::X, &[0])).unwrap();
    b.apply(&g(Gate::X, &[1])).unwrap();
    b.apply(&Instruction::Reset { qubit: 0 }).unwrap();
    b.apply(&Instruction::Measure {
        qubit: 0,
        classical_bit: 0,
    })
    .unwrap();
    b.apply(&Instruction::Measure {
        qubit: 1,
        classical_bit: 1,
    })
    .unwrap();
    let bits = b.classical_results();
    assert_eq!(bits, &[false, true], "reset then measure");
}

#[test]
fn stabilizer_gpu_gates_then_measure_matches_cpu() {
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };
    // Same circuit, same seed → CPU and GPU must agree on every classical bit.
    let circuit: Vec<Instruction> = vec![
        g(Gate::H, &[0]),
        g(Gate::Cx, &[0, 1]),
        g(Gate::H, &[2]),
        g(Gate::Cx, &[2, 3]),
        g(Gate::Cz, &[1, 2]),
        Instruction::Measure {
            qubit: 0,
            classical_bit: 0,
        },
        Instruction::Measure {
            qubit: 1,
            classical_bit: 1,
        },
        Instruction::Measure {
            qubit: 2,
            classical_bit: 2,
        },
        Instruction::Measure {
            qubit: 3,
            classical_bit: 3,
        },
    ];

    let mut cpu = StabilizerBackend::new(42);
    cpu.init(4, 4).unwrap();
    for inst in &circuit {
        cpu.apply(inst).unwrap();
    }

    let mut gpu = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(4, 4).unwrap();
    for inst in &circuit {
        gpu.apply(inst).unwrap();
    }

    assert_eq!(
        cpu.classical_results(),
        gpu.classical_results(),
        "classical bits must match"
    );
}

#[test]
fn stabilizer_gpu_export_statevector_matches_cpu() {
    use prism_q::StabilizerBackend;

    let Some(f) = Fixture::try_new() else { return };
    let circuit = &[g(Gate::H, &[0]), g(Gate::Cx, &[0, 1]), g(Gate::H, &[2])];

    let mut cpu = StabilizerBackend::new(42);
    cpu.init(3, 0).unwrap();
    for inst in circuit {
        cpu.apply(inst).unwrap();
    }
    let cpu_sv = cpu.export_statevector().unwrap();

    let mut gpu = StabilizerBackend::new(42).with_gpu(f.ctx.clone());
    gpu.init(3, 0).unwrap();
    for inst in circuit {
        gpu.apply(inst).unwrap();
    }
    let gpu_sv = gpu.export_statevector().unwrap();

    assert_eq!(cpu_sv.len(), gpu_sv.len());
    for (i, (c, g_)) in cpu_sv.iter().zip(gpu_sv.iter()).enumerate() {
        assert!((c - g_).norm() < 1e-12, "amp[{i}]: cpu={c:?}, gpu={g_:?}");
    }
}

// ============================================================================
// Stabilizer GPU public dispatch
// ============================================================================

/// `run_with_stabilizer_gpu` routes through `BackendKind::StabilizerGpu` and the
/// crossover. For small n the crossover sends the circuit to the CPU
/// stabilizer automatically. Verifies end-to-end dispatch matches a direct
/// `BackendKind::Stabilizer` run.
#[test]
fn run_with_stabilizer_gpu_matches_cpu_below_threshold() {
    use prism_q::{run_with, run_with_stabilizer_gpu, BackendKind, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    let mut circuit = Circuit::new(8, 8);
    circuit.add_gate(Gate::H, &[0]);
    for i in 0..7 {
        circuit.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..8 {
        circuit.add_measure(i, i);
    }

    let cpu = run_with(BackendKind::Stabilizer, &circuit, 42).unwrap();
    let gpu = run_with_stabilizer_gpu(&circuit, 42, f.ctx.clone()).unwrap();
    assert_eq!(cpu.classical_bits, gpu.classical_bits);
}

/// End-to-end dispatch check at a large qubit count. With the default
/// crossover (100_000) this routes to CPU underneath, so the comparison is
/// ultimately CPU-vs-CPU at the same seed. When future milestones lower the
/// threshold the same test exercises the real GPU path and the assertion
/// still holds.
#[test]
fn run_with_stabilizer_gpu_matches_cpu_at_scale() {
    use prism_q::{run_with, run_with_stabilizer_gpu, BackendKind};

    let Some(f) = Fixture::try_new() else { return };

    let n = 600;
    let circuit = prism_q::circuits::clifford_heavy_circuit(n, 3, 0xC1FD);
    // Add measurements so classical_bits is populated.
    let mut circuit = circuit;
    circuit.num_classical_bits = n;
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    let cpu = run_with(BackendKind::Stabilizer, &circuit, 42).unwrap();
    let gpu = run_with_stabilizer_gpu(&circuit, 42, f.ctx.clone()).unwrap();
    assert_eq!(
        cpu.classical_bits, gpu.classical_bits,
        "CPU and GPU stabilizer measurement bits must agree"
    );
}

// ============================================================================
// GPU BTS sampling
// ============================================================================

/// `run_shots_compiled_with_gpu` routes BTS sampling through the GPU when the
/// circuit compiles to a flat sparse parity. Statistical test: 50 Bell pairs
/// measured per shot. Every shot must have each pair correlated
/// (outcome_{2i} == outcome_{2i+1}). The correlation is deterministic, so any
/// bit error in the GPU path surfaces.
#[test]
fn run_shots_compiled_with_gpu_bell_pairs_are_correlated() {
    use prism_q::{run_shots_compiled_with_gpu, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    let n_pairs = 50;
    let n = 2 * n_pairs;
    let num_shots = 4000;
    let mut circuit = Circuit::new(n, n);
    for p in 0..n_pairs {
        circuit.add_gate(Gate::H, &[2 * p]);
        circuit.add_gate(Gate::Cx, &[2 * p, 2 * p + 1]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }

    let result = run_shots_compiled_with_gpu(&circuit, num_shots, 42, f.ctx.clone()).unwrap();
    assert_eq!(result.shots.len(), num_shots);
    for (s, shot) in result.shots.iter().enumerate() {
        assert_eq!(shot.len(), n, "shot {s} has wrong bit count");
        for p in 0..n_pairs {
            assert_eq!(
                shot[2 * p],
                shot[2 * p + 1],
                "shot {s} pair {p}: bell pair bits must match, got [{}, {}]",
                shot[2 * p],
                shot[2 * p + 1]
            );
        }
    }
}

/// Statistical equivalence: CPU BTS (`run_shots_compiled`) and GPU BTS
/// (`run_shots_compiled_with_gpu`) draw from the same distribution. With the
/// same seed and a flat sparse parity their marginal per-measurement bit
/// rates should agree to within a few sigma over 10_000 shots.
#[test]
fn run_shots_compiled_with_gpu_distribution_matches_cpu() {
    use prism_q::{run_shots_compiled, run_shots_compiled_with_gpu, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    // Independent random bits: H on each qubit then measure. Each qubit's
    // marginal should be ~0.5. Use a modest qubit count so CPU reference
    // computes quickly.
    let n = 32;
    let num_shots = 10_000;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }

    let cpu = run_shots_compiled(&circuit, num_shots, 42).unwrap();
    let gpu = run_shots_compiled_with_gpu(&circuit, num_shots, 42, f.ctx.clone()).unwrap();

    assert_eq!(cpu.shots.len(), gpu.shots.len());
    let cpu_marginals: Vec<f64> = (0..n)
        .map(|q| cpu.shots.iter().filter(|s| s[q]).count() as f64 / num_shots as f64)
        .collect();
    let gpu_marginals: Vec<f64> = (0..n)
        .map(|q| gpu.shots.iter().filter(|s| s[q]).count() as f64 / num_shots as f64)
        .collect();
    // Both should be ~0.5. Allow 5 sigma for 10k shots: sigma ~ 0.005.
    for q in 0..n {
        assert!(
            (cpu_marginals[q] - 0.5).abs() < 0.025,
            "cpu marginal[{q}] = {} (expected ~0.5)",
            cpu_marginals[q]
        );
        assert!(
            (gpu_marginals[q] - 0.5).abs() < 0.025,
            "gpu marginal[{q}] = {} (expected ~0.5)",
            gpu_marginals[q]
        );
    }
}

/// `run_shots_with(BackendKind::StabilizerGpu)` should route Clifford shot
/// sampling through the same compiled GPU sampler as
/// `run_shots_compiled_with_gpu`, not the raw tableau measurement loop.
#[test]
fn run_shots_with_stabilizer_gpu_matches_compiled_gpu_sampling() {
    use prism_q::{run_shots_compiled_with_gpu, run_shots_with, BackendKind, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    let n = 32;
    let num_shots = 4096;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }

    let compiled = run_shots_compiled_with_gpu(&circuit, num_shots, 42, f.ctx.clone()).unwrap();
    let explicit = run_shots_with(
        BackendKind::StabilizerGpu {
            context: f.ctx.clone(),
        },
        &circuit,
        num_shots,
        42,
    )
    .unwrap();

    assert_eq!(explicit.shots, compiled.shots);
}

/// Reused GPU BTS scratch should not change the packed output stream for a
/// fixed sequence of batch requests. Two samplers with the same seed and chunk
/// schedule should produce identical packed batches, even though one sampler
/// reuses its GPU BTS cache across calls.
#[test]
fn run_shots_compiled_with_gpu_repeated_batches_match_fresh_sampler() {
    use prism_q::{compile_measurements, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    let n = 128;
    let batch_shots = std::env::var("PRISM_GPU_BTS_MIN_SHOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(prism_q::gpu::BTS_MIN_SHOTS_DEFAULT);
    if batch_shots == 0 || batch_shots > 262_144 {
        return;
    }
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }

    let mut reused = compile_measurements(&circuit, 42)
        .unwrap()
        .with_gpu(f.ctx.clone());
    let mut fresh = compile_measurements(&circuit, 42)
        .unwrap()
        .with_gpu(f.ctx.clone());

    let reused_first = reused.sample_bulk_packed(batch_shots).raw_data().to_vec();
    let fresh_first = fresh.sample_bulk_packed(batch_shots).raw_data().to_vec();
    assert_eq!(reused_first, fresh_first);

    let reused_second = reused.sample_bulk_packed(batch_shots).raw_data().to_vec();
    let fresh_second = fresh.sample_bulk_packed(batch_shots).raw_data().to_vec();
    assert_eq!(reused_second, fresh_second);
}

#[test]
fn sample_bulk_packed_device_to_host_is_reproducible() {
    use prism_q::{compile_measurements, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    let shots = std::env::var("PRISM_GPU_BTS_MIN_SHOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(prism_q::gpu::BTS_MIN_SHOTS_DEFAULT);
    if shots == 0 || shots > 262_144 {
        return;
    }

    let n = 128;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }

    let mut gpu_a = compile_measurements(&circuit, 42)
        .unwrap()
        .with_gpu(f.ctx.clone());
    let copied_a = gpu_a
        .sample_bulk_packed_device(shots)
        .unwrap()
        .to_host()
        .unwrap();

    let mut gpu_b = compile_measurements(&circuit, 42)
        .unwrap()
        .with_gpu(f.ctx.clone());
    let copied_b = gpu_b
        .sample_bulk_packed_device(shots)
        .unwrap()
        .to_host()
        .unwrap();

    assert_eq!(copied_a.raw_data(), copied_b.raw_data());
}

#[test]
fn sample_bulk_packed_device_marginals_match_host_reduction() {
    use prism_q::{compile_measurements, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    let shots = std::env::var("PRISM_GPU_BTS_MIN_SHOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(prism_q::gpu::BTS_MIN_SHOTS_DEFAULT);
    if shots == 0 || shots > 262_144 {
        return;
    }

    let n = 128;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }

    let mut gpu = compile_measurements(&circuit, 42)
        .unwrap()
        .with_gpu(f.ctx.clone());
    let device = gpu.sample_bulk_packed_device(shots).unwrap();
    let device_marginals = device.marginals().unwrap();
    let host = device.to_host().unwrap();

    let host_marginals: Vec<f64> = (0..n)
        .map(|m| {
            let mut ones = 0u64;
            for s in 0..shots {
                if host.get_bit(s, m) {
                    ones += 1;
                }
            }
            ones as f64 / shots as f64
        })
        .collect();

    assert_eq!(device_marginals, host_marginals);
}

#[test]
fn sample_bulk_packed_device_counts_match_host_counts() {
    use prism_q::{compile_measurements, Circuit};

    let Some(f) = Fixture::try_new() else { return };

    let shots = std::env::var("PRISM_GPU_BTS_MIN_SHOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(prism_q::gpu::BTS_MIN_SHOTS_DEFAULT)
        .max(131_072);
    if shots > 262_144 {
        return;
    }

    let n = 128;
    let rank = 12;
    let mut circuit = Circuit::new(n, n);
    for q in 0..rank {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }

    let mut gpu = compile_measurements(&circuit, 42)
        .unwrap()
        .with_gpu(f.ctx.clone());
    let device = gpu.sample_bulk_packed_device(shots).unwrap();
    let device_counts = device.counts().unwrap();
    let host_counts = device.to_host().unwrap().counts();

    assert_eq!(device_counts, host_counts);
}
