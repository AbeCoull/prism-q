//! Device against host for the density matrix: every channel, terminal, and
//! gate shape runs on both, and the full `4^n` buffer must agree within 1e-12.
//! A diagonal sandwich leaves every diagonal entry of rho alone, so a
//! probabilities comparison would pass on a broken arm; the buffer is what is
//! compared. Without a usable GPU every test returns before asserting; set
//! `PRISM_REQUIRE_GPU=1` to make that a hard failure.

#![cfg(feature = "gpu")]

mod common;

use std::sync::Arc;

use common::{SEED, all_pauli_masks, assert_probs_close};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::circuit::fusion::fuse_circuit_for_width;
use prism_q::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec, smallvec};
use prism_q::gates::{
    BatchPhaseData, BatchRzzData, DiagEntry, DiagonalBatchData, Gate, McuData, Multi2qData,
    MultiFusedData,
};
use prism_q::gpu::GpuContext;
use prism_q::{
    BackendKind, NoiseChannel, NoiseEvent, NoiseModel, PauliTerm, Placement, circuits, sim,
};

const EPS: f64 = 1e-12;

struct Fixture {
    ctx: Arc<GpuContext>,
}

impl Fixture {
    fn try_new() -> Option<Self> {
        match GpuContext::new(0) {
            Ok(ctx) => Some(Self { ctx }),
            Err(e) => {
                assert!(
                    std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                    "PRISM_REQUIRE_GPU is set but no usable GPU was found ({e}). \
                     Unset it to allow this suite to skip."
                );
                eprintln!("SKIP: no usable GPU ({e})");
                None
            }
        }
    }

    fn kind(&self) -> BackendKind {
        BackendKind::DensityMatrixGpu {
            context: self.ctx.clone(),
        }
    }

    fn device_backend(&self) -> DensityMatrixBackend {
        DensityMatrixBackend::new(SEED).with_gpu(self.ctx.clone())
    }

    /// Host and device backends holding the same mixed `n`-qubit state: a random
    /// circuit followed by amplitude damping, so every entry class of rho is
    /// populated and the off-diagonals are not a pure state's.
    fn prepared_pair(&self, n: usize) -> (DensityMatrixBackend, DensityMatrixBackend) {
        let circuit = circuits::random_circuit(n, 4, SEED);
        let mut cpu = DensityMatrixBackend::new(SEED);
        let mut gpu = self.device_backend();
        for backend in [&mut cpu, &mut gpu] {
            backend.init(n, n).unwrap();
            backend.apply_instructions(&circuit.instructions).unwrap();
            backend.apply_1q_kraus(n - 2, &amplitude_damping(0.3));
        }
        (cpu, gpu)
    }
}

fn assert_same_mixture(cpu: &DensityMatrixBackend, gpu: &DensityMatrixBackend, label: &str) {
    assert_eq!(
        cpu.classical_results(),
        gpu.classical_results(),
        "{label}: classical bits mismatch"
    );
    let want = cpu.density_matrix().unwrap();
    let got = gpu.density_matrix().unwrap();
    assert_eq!(want.len(), got.len(), "{label}: buffer length mismatch");
    for (i, (c, g)) in want.iter().zip(&got).enumerate() {
        let diff = (c - g).norm();
        assert!(
            diff < EPS,
            "{label}: entry {i} cpu={c:?} gpu={g:?} |diff|={diff}"
        );
    }
}

fn g(gate: Gate, targets: &[usize]) -> Instruction {
    let mut tv: SmallVec<[usize; 4]> = smallvec![];
    tv.extend_from_slice(targets);
    Instruction::Gate { gate, targets: tv }
}

fn c(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

fn amplitude_damping(gamma: f64) -> Vec<[[Complex64; 2]; 2]> {
    let zero = c(0.0);
    vec![
        [[c(1.0), zero], [zero, c((1.0 - gamma).sqrt())]],
        [[zero, c(gamma.sqrt())], [zero, zero]],
    ]
}

fn depolarizing_1q(p: f64) -> Vec<[[Complex64; 2]; 2]> {
    let zero = c(0.0);
    let w = c((p / 3.0).sqrt());
    let iw = Complex64::new(0.0, (p / 3.0).sqrt());
    vec![
        [[c((1.0 - p).sqrt()), zero], [zero, c((1.0 - p).sqrt())]],
        [[zero, w], [w, zero]],
        [[zero, -iw], [iw, zero]],
        [[w, zero], [zero, -w]],
    ]
}

/// Correlated `ZZ` dephasing: compiles to a diagonal superoperator.
fn correlated_zz(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let zero = c(0.0);
    let diag = |scale: f64, signs: [f64; 4]| {
        let mut m = [[zero; 4]; 4];
        for (t, sign) in signs.iter().enumerate() {
            m[t][t] = c(scale * sign);
        }
        m
    };
    vec![
        diag((1.0 - p).sqrt(), [1.0, 1.0, 1.0, 1.0]),
        diag(p.sqrt(), [1.0, -1.0, -1.0, 1.0]),
    ]
}

/// `{sqrt(1-p) I, sqrt(p) X(x)Z}`: fills the 16x16 superoperator.
fn h_conjugated_zz(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let zero = c(0.0);
    let mut k0 = [[zero; 4]; 4];
    let mut k1 = [[zero; 4]; 4];
    for t in 0..4 {
        k0[t][t] = c((1.0 - p).sqrt());
        let sign = if t & 1 == 0 { 1.0 } else { -1.0 };
        k1[t][t ^ 2] = c(p.sqrt() * sign);
    }
    vec![k0, k1]
}

fn sample_2x2() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.6, -0.1), Complex64::new(-0.3, 0.2)],
        [Complex64::new(0.2, 0.4), Complex64::new(0.7, -0.2)],
    ]
}

fn sample_4x4() -> [[Complex64; 4]; 4] {
    let mut mat = [[c(0.0); 4]; 4];
    for (r, row) in mat.iter_mut().enumerate() {
        for (col, entry) in row.iter_mut().enumerate() {
            *entry = Complex64::new(0.1 * (r as f64 + 1.0), 0.07 * (col as f64 + 1.0));
        }
    }
    mat
}

fn pauli_rot_sample() -> Gate {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_pauli_rotation(0.53, &[PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)]);
    match &circuit.instructions[0] {
        Instruction::Gate { gate, .. } => gate.clone(),
        _ => unreachable!("add_pauli_rotation appends a gate"),
    }
}

/// One instruction per `Gate` variant on a six-qubit register.
fn gate_instructions() -> Vec<Instruction> {
    let m2 = sample_2x2();
    let m4 = sample_4x4();
    vec![
        g(Gate::Id, &[0]),
        g(Gate::X, &[1]),
        g(Gate::Y, &[2]),
        g(Gate::Z, &[3]),
        g(Gate::H, &[4]),
        g(Gate::S, &[5]),
        g(Gate::Sdg, &[0]),
        g(Gate::T, &[1]),
        g(Gate::Tdg, &[2]),
        g(Gate::SX, &[3]),
        g(Gate::SXdg, &[4]),
        g(Gate::Rx(0.37), &[5]),
        g(Gate::Ry(1.11), &[0]),
        g(Gate::Rz(-0.62), &[1]),
        g(Gate::P(0.44), &[2]),
        g(Gate::Fused(Box::new(m2)), &[3]),
        g(Gate::Rzz(0.3), &[0, 5]),
        g(Gate::Cx, &[1, 4]),
        g(Gate::Cz, &[2, 3]),
        g(Gate::Swap, &[0, 3]),
        g(Gate::Cu(Box::new(m2)), &[4, 1]),
        g(
            Gate::Mcu(Box::new(McuData {
                mat: m2,
                num_controls: 2,
            })),
            &[0, 2, 5],
        ),
        g(
            Gate::BatchPhase(Box::new(BatchPhaseData {
                phases: smallvec![
                    (2usize, Complex64::from_polar(1.0, 0.5)),
                    (4usize, Complex64::from_polar(1.0, -1.3))
                ],
            })),
            &[1],
        ),
        g(
            Gate::BatchRzz(Box::new(BatchRzzData {
                edges: vec![(0, 1, 0.3), (2, 5, 0.5), (3, 4, -0.9)],
            })),
            &[0, 1, 2, 3, 4, 5],
        ),
        g(
            Gate::DiagonalBatch(Box::new(DiagonalBatchData {
                entries: vec![
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
                        q0: 3,
                        q1: 5,
                        same: Complex64::from_polar(1.0, 0.1),
                        diff: Complex64::from_polar(1.0, -0.4),
                    },
                ],
            })),
            &[0, 1, 2, 3, 5],
        ),
        g(
            Gate::MultiFused(Box::new(MultiFusedData {
                gates: vec![(0, m2), (3, m2), (5, m2)],
                all_diagonal: false,
            })),
            &[0, 1, 2, 3, 4, 5],
        ),
        g(Gate::Fused2q(Box::new(m4)), &[2, 4]),
        g(
            Gate::Multi2q(Box::new(Multi2qData {
                gates: vec![(0, 1, m4), (3, 5, m4)],
            })),
            &[0, 1, 3, 5],
        ),
        g(Gate::QftBlock { start: 1, num: 4 }, &[1, 2, 3, 4]),
        g(pauli_rot_sample(), &[0, 2, 5]),
    ]
}

#[test]
fn dm_gpu_every_gate_variant_matches_cpu() {
    let Some(f) = Fixture::try_new() else { return };
    let (mut cpu, mut gpu) = f.prepared_pair(6);
    for inst in gate_instructions() {
        cpu.apply(&inst).unwrap();
        gpu.apply(&inst).unwrap();
        assert_same_mixture(&cpu, &gpu, &format!("after {inst:?}"));
    }
}

type Step = Box<dyn Fn(&mut DensityMatrixBackend)>;

#[test]
fn dm_gpu_channel_kernels_match_cpu() {
    let Some(f) = Fixture::try_new() else { return };
    for n in [8usize, 10] {
        let (mut cpu, mut gpu) = f.prepared_pair(n);
        let steps: Vec<(&str, Step)> = vec![
            (
                "amp damp q0",
                Box::new(|b| b.apply_1q_kraus(0, &amplitude_damping(0.05))),
            ),
            (
                "depolarizing 1q qtop",
                Box::new(move |b| b.apply_1q_kraus(n - 1, &depolarizing_1q(0.02))),
            ),
            (
                "depolarizing 2q (0, qtop)",
                Box::new(move |b| b.apply_2q_depolarizing(0, n - 1, 0.02)),
            ),
            (
                "depolarizing 2q (3, 1)",
                Box::new(|b| b.apply_2q_depolarizing(3, 1, 0.07)),
            ),
            (
                "diagonal 2q kraus (2, 3)",
                Box::new(|b| b.apply_2q_kraus(2, 3, &correlated_zz(0.02))),
            ),
            (
                "diagonal 2q kraus (0, qtop)",
                Box::new(move |b| b.apply_2q_kraus(0, n - 1, &correlated_zz(0.02))),
            ),
            (
                "dense 2q kraus (2, 3)",
                Box::new(|b| b.apply_2q_kraus(2, 3, &h_conjugated_zz(0.02))),
            ),
            (
                "dense 2q kraus (0, qtop)",
                Box::new(move |b| b.apply_2q_kraus(0, n - 1, &h_conjugated_zz(0.02))),
            ),
            (
                "dense 2q kraus (qtop, 1)",
                Box::new(move |b| b.apply_2q_kraus(n - 1, 1, &h_conjugated_zz(0.05))),
            ),
        ];
        for (label, step) in steps {
            step(&mut cpu);
            step(&mut gpu);
            assert_same_mixture(&cpu, &gpu, &format!("{n}q {label}"));
        }
    }
}

/// A model that fires every `NoiseChannel` variant, one after each of the
/// first eight gates, with a two-qubit channel wherever the gate has a pair.
fn every_channel_model(circuit: &Circuit) -> NoiseModel {
    let kinds: Vec<NoiseChannel> = vec![
        NoiseChannel::Pauli {
            px: 0.01,
            py: 0.02,
            pz: 0.03,
        },
        NoiseChannel::Depolarizing { p: 0.04 },
        NoiseChannel::AmplitudeDamping { gamma: 0.05 },
        NoiseChannel::PhaseDamping { gamma: 0.06 },
        NoiseChannel::ThermalRelaxation {
            t1: 50.0,
            t2: 30.0,
            gate_time: 1.0,
        },
        NoiseChannel::Custom {
            kraus: amplitude_damping(0.07),
        },
        NoiseChannel::TwoQubitDepolarizing { p: 0.03 },
        NoiseChannel::Kraus2q {
            kraus: h_conjugated_zz(0.04),
        },
    ];
    let mut next = 0usize;
    let after_gate = circuit
        .instructions
        .iter()
        .map(|inst| {
            let Instruction::Gate { targets, .. } = inst else {
                return Vec::new();
            };
            let mut events = Vec::new();
            if let Some(channel) = kinds.get(next).cloned() {
                if channel.num_qubits() == 1 {
                    next += 1;
                    events.push(NoiseEvent {
                        channel,
                        qubits: [targets[0]].into_iter().collect(),
                    });
                } else if targets.len() == 2 {
                    next += 1;
                    events.push(NoiseEvent {
                        channel,
                        qubits: [targets[0], targets[1]].into_iter().collect(),
                    });
                }
            }
            events
        })
        .collect();
    assert_eq!(
        next,
        kinds.len(),
        "the fixture must place every channel kind"
    );
    NoiseModel {
        after_gate,
        readout: vec![None; circuit.num_classical_bits],
    }
}

fn single_qubit_observables(n: usize) -> Vec<Vec<PauliTerm>> {
    let mut observables = Vec::new();
    for q in 0..n {
        observables.push(vec![PauliTerm::x(q)]);
        observables.push(vec![PauliTerm::y(q)]);
        observables.push(vec![PauliTerm::z(q)]);
    }
    observables.push(vec![PauliTerm::x(0), PauliTerm::y(n - 1)]);
    observables.push(vec![PauliTerm::y(1), PauliTerm::z(2), PauliTerm::x(3)]);
    observables
}

#[test]
fn dm_gpu_every_noise_channel_kind_matches_cpu_through_simulate() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 8;
    let unitary = circuits::random_circuit(n, 3, SEED);
    let unitary_noise = every_channel_model(&unitary);
    let mut circuit = unitary.clone();
    circuit.measure_all();
    let mut noise = every_channel_model(&unitary);
    noise
        .after_gate
        .resize(circuit.instructions.len(), Vec::new());
    noise.readout = vec![None; circuit.num_classical_bits];

    let host = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .run()
        .unwrap();
    let device = sim::simulate(&circuit)
        .backend(f.kind())
        .noise(&noise)
        .seed(SEED)
        .run()
        .unwrap();
    assert_eq!(device.metadata.placement, Placement::Device);
    assert_probs_close(
        &device.probabilities.unwrap().to_vec(),
        &host.probabilities.unwrap().to_vec(),
        EPS,
        "every channel kind, probabilities",
    );

    let observables = single_qubit_observables(n);
    let host = sim::simulate(&unitary)
        .backend(BackendKind::DensityMatrix)
        .noise(&unitary_noise)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    let device = sim::simulate(&unitary)
        .backend(f.kind())
        .noise(&unitary_noise)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    for (k, (h, d)) in host.iter().zip(&device).enumerate() {
        assert!(
            (h - d).abs() < EPS,
            "every channel kind, observable {k}: host {h} device {d}"
        );
    }
}

#[test]
fn dm_gpu_measure_reset_and_conditional_match_cpu() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 8;
    let (mut cpu, mut gpu) = f.prepared_pair(n);
    let steps = [
        Instruction::Measure {
            qubit: 0,
            classical_bit: 0,
        },
        Instruction::Reset { qubit: 1 },
        Instruction::Conditional {
            condition: ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: smallvec![2],
        },
        Instruction::Conditional {
            condition: ClassicalCondition::BitIsZero(0),
            gate: Gate::H,
            targets: smallvec![3],
        },
        g(Gate::H, &[n - 1]),
        Instruction::Measure {
            qubit: n - 1,
            classical_bit: 1,
        },
        Instruction::Measure {
            qubit: 4,
            classical_bit: 2,
        },
        Instruction::Reset { qubit: n - 1 },
        Instruction::Reset { qubit: 0 },
        g(Gate::Rx(0.7), &[0]),
        Instruction::Measure {
            qubit: 0,
            classical_bit: 3,
        },
    ];
    for inst in steps {
        cpu.apply(&inst).unwrap();
        gpu.apply(&inst).unwrap();
        assert_same_mixture(&cpu, &gpu, &format!("after {inst:?}"));
    }
    assert!(
        cpu.classical_results().iter().any(|&b| b),
        "the fixture must record at least one 1 outcome"
    );
}

fn count_gates(circuit: &Circuit, want: impl Fn(&Gate) -> bool) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Gate { gate, .. } if want(gate)))
        .count()
}

#[test]
fn dm_gpu_fused_stream_matches_unfused_cpu() {
    // The device path serves the same fused payloads the host accepts, and the
    // payloads are asserted so a fusion change cannot quietly drain the test.
    let Some(f) = Fixture::try_new() else { return };
    const N: usize = 8;
    for (label, circuit, want) in [
        (
            "qaoa",
            circuits::qaoa_circuit(N, 3, SEED),
            &[("BatchRzz", 1usize), ("DiagonalBatch", 0)][..],
        ),
        (
            "diagonal mixed",
            circuits::diagonal_mixed_circuit(N, 3, SEED),
            &[("BatchRzz", 1), ("DiagonalBatch", 1)][..],
        ),
        (
            "random",
            circuits::random_circuit(N, 4, SEED),
            &[("Multi2q", 1), ("MultiFused", 1)][..],
        ),
    ] {
        let fused = fuse_circuit_for_width(&circuit, true, 2 * N).into_owned();
        for &(form, count) in want {
            let got = count_gates(&fused, |gate| match form {
                "BatchRzz" => matches!(gate, Gate::BatchRzz(_)),
                "DiagonalBatch" => matches!(gate, Gate::DiagonalBatch(_)),
                "Multi2q" => matches!(gate, Gate::Multi2q(_)),
                "MultiFused" => matches!(gate, Gate::MultiFused(_)),
                _ => unreachable!(),
            });
            assert!(got >= count, "{label}: expected {count} {form}, got {got}");
        }
        let mut cpu = DensityMatrixBackend::new(SEED);
        cpu.init(N, 0).unwrap();
        cpu.apply_instructions(&circuit.instructions).unwrap();
        let mut gpu = f.device_backend();
        gpu.init(N, 0).unwrap();
        gpu.apply_instructions(&fused.instructions).unwrap();
        assert_same_mixture(
            &cpu,
            &gpu,
            &format!("{label} fused on device vs unfused on host"),
        );

        let outcome = sim::simulate(&circuit)
            .backend(f.kind())
            .seed(SEED)
            .run()
            .unwrap();
        assert_eq!(outcome.metadata.placement, Placement::Device);
        assert_probs_close(
            &outcome.probabilities.unwrap().to_vec(),
            &cpu.probabilities().unwrap(),
            EPS,
            &format!("{label} dispatched device run"),
        );
    }
}

#[test]
fn dm_gpu_terminals_match_cpu() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 5;
    let (cpu, gpu) = f.prepared_pair(n);

    assert_probs_close(
        &gpu.probabilities().unwrap(),
        &cpu.probabilities().unwrap(),
        EPS,
        "probabilities",
    );
    assert!(
        (cpu.purity() - gpu.purity()).abs() < EPS,
        "purity: cpu {} gpu {}",
        cpu.purity(),
        gpu.purity()
    );
    assert!(cpu.purity() < 0.999, "the fixture must be mixed");

    let masks = all_pauli_masks(n);
    let want = cpu.expectations_pauli(&masks);
    let got = gpu.expectations_pauli(&masks);
    for (k, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!((w - g).abs() < EPS, "pauli {:?}: cpu {w} gpu {g}", masks[k]);
    }
    let observables = single_qubit_observables(n);
    let want = cpu.pauli_expectations(&observables).unwrap();
    let got = gpu.pauli_expectations(&observables).unwrap();
    for (k, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!((w - g).abs() < EPS, "observable {k}: cpu {w} gpu {g}");
    }

    for q in 0..n {
        let want = cpu.reduced_density_matrix_1q(q).unwrap();
        let got = gpu.reduced_density_matrix_1q(q).unwrap();
        for r in 0..2 {
            for col in 0..2 {
                assert!(
                    (want[r][col] - got[r][col]).norm() < EPS,
                    "rdm q{q} [{r}][{col}]: cpu {:?} gpu {:?}",
                    want[r][col],
                    got[r][col]
                );
            }
        }
        let want = cpu.qubit_probability(q).unwrap();
        let got = gpu.qubit_probability(q).unwrap();
        assert!(
            (want - got).abs() < EPS,
            "qubit_probability q{q}: cpu {want} gpu {got}"
        );
    }
}

#[test]
fn dm_gpu_init_from_amplitudes_matches_cpu() {
    let Some(f) = Fixture::try_new() else { return };
    let n = 6;
    let mut amps: Vec<Complex64> = (0..1usize << n)
        .map(|i| Complex64::new(0.3 + 0.01 * i as f64, -0.2 + 0.02 * (i % 7) as f64))
        .collect();
    let norm = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    for a in &mut amps {
        *a /= norm;
    }
    let mut cpu = DensityMatrixBackend::new(SEED);
    let mut gpu = f.device_backend();
    cpu.init_from_amplitudes(amps.clone(), 0).unwrap();
    gpu.init_from_amplitudes(amps, 0).unwrap();
    assert_same_mixture(&cpu, &gpu, "outer product");
    for inst in [
        g(Gate::H, &[2]),
        g(Gate::Cx, &[2, 5]),
        g(Gate::Rzz(0.4), &[0, 4]),
    ] {
        cpu.apply(&inst).unwrap();
        gpu.apply(&inst).unwrap();
    }
    assert_same_mixture(&cpu, &gpu, "gates after a start state");
}

#[test]
fn dm_gpu_over_budget_init_names_the_mixture() {
    let Some(f) = Fixture::try_new() else { return };
    let over = f.ctx.max_qubits_for_statevector().unwrap() / 2 + 1;
    let mut backend = f.device_backend();
    match backend.init(over, 0).unwrap_err() {
        prism_q::PrismError::IncompatibleBackend { backend, reason } => {
            assert_eq!(backend, "density_matrix-gpu");
            assert!(
                reason.contains("free on the GPU") && reason.contains("4^n"),
                "expected the mixture budget message, got: {reason}"
            );
            assert!(reason.contains(&format!("{over} qubits")));
        }
        other => panic!("expected the device budget error, got {other:?}"),
    }
}

#[test]
fn dm_gpu_kind_is_explicit_only() {
    // The noisy terminals accept only the two mixture kinds, so an AutoGpu run
    // under noise is rejected rather than routed to the device mixture.
    let Some(f) = Fixture::try_new() else { return };
    let mut circuit = circuits::random_circuit(6, 2, SEED);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let err = sim::simulate(&circuit)
        .backend(BackendKind::AutoGpu {
            context: f.ctx.clone(),
        })
        .noise(&noise)
        .seed(SEED)
        .run()
        .unwrap_err();
    assert!(
        err.to_string().contains("density-matrix backend"),
        "unexpected rejection: {err}"
    );

    let device = sim::simulate(&circuit)
        .backend(f.kind())
        .noise(&noise)
        .seed(SEED)
        .shots(64)
        .unwrap();
    assert_eq!(device.metadata.placement, Placement::Device);
    let host = sim::simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .shots(64)
        .unwrap();
    assert_eq!(
        device.counts(),
        host.counts(),
        "shots drawn from the same distribution"
    );
}
