//! Macrobenchmarks: circuit family sweeps (qubit count and depth).
//!
//! Use `--features bench-fast` for a quick run that reduces warmup and
//! measurement time. Omit for the full suite with default Criterion timing.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
#[cfg(feature = "bench-internal")]
use prism_q::backend::tensornetwork::scalar_expectation;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::circuit::{Circuit, SmallVec};
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::{
    BackendKind, ClassicalCondition, Instruction, MpsBackend, Parameters, PauliObservable,
    PauliTerm, PreparedCircuit, run_expectation_gradient, run_expectation_gradient_shift,
    run_expectation_values, run_observable_expectation,
};
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Duration;

mod common;
use common::{SEED, configure_group, run_shots_with, run_with};

fn run_shots_with_noise(
    kind: BackendKind,
    circuit: &Circuit,
    noise: &prism_q::NoiseModel,
    num_shots: usize,
    seed: u64,
) -> prism_q::Result<prism_q::ShotsResult> {
    sim::simulate(circuit)
        .backend(kind)
        .noise(noise)
        .seed(seed)
        .shots(num_shots)
}

/// Statevector rows above 26q, opted in via `PRISM_BENCH_HIGH_QUBITS=<max>`.
///
/// Off by default: a dense `.run()` holds the statevector and its readback
/// probability vector at once, so 28q peaks near 6 GiB and 30q near 24 GiB.
/// `<max>` bounds the largest row a host admits.
fn high_qubit_sizes() -> Vec<usize> {
    let cap: usize = std::env::var("PRISM_BENCH_HIGH_QUBITS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    [28, 30].into_iter().filter(|&n| n <= cap).collect()
}

// ---- Bench-specific circuit builders (not shared) ----

fn qft_like_circuit(n_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for i in 0..n_qubits {
        circuit.add_gate(Gate::H, &[i]);
        for j in (i + 1)..n_qubits {
            let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
            circuit.add_gate(Gate::cphase(theta), &[i, j]);
        }
    }

    circuit
}

fn sparse_entanglement_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for _ in 0..depth {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
        }
        if n_qubits >= 2 {
            circuit.add_gate(Gate::Cx, &[0, n_qubits - 1]);
        }
    }

    circuit
}

fn dense_entanglement_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for _ in 0..depth {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
        }
        for q in 0..n_qubits - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    circuit
}

fn random_clifford_circuit(n_qubits: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];

    for layer in 0..depth {
        for q in 0..n_qubits {
            let gate_idx = rng.random_range(0..cliffords.len());
            circuit.add_gate(cliffords[gate_idx].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n_qubits - 1).step_by(2) {
            if rng.random_bool(0.5) {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    circuit
}

fn random_single_qubit_circuit(n_qubits: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let gates = [Gate::H, Gate::X, Gate::Y, Gate::Z, Gate::S, Gate::T];

    for _ in 0..depth {
        for q in 0..n_qubits {
            let gate_idx = rng.random_range(0..gates.len());
            circuit.add_gate(gates[gate_idx].clone(), &[q]);
        }
    }
    circuit
}

fn mps_adjacent_phase_circuit(n_qubits: usize, rounds: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }

    for _ in 0..rounds {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::Ry(0.17 + q as f64 * 0.003), &[q]);
        }
        for q in (0..n_qubits.saturating_sub(1)).step_by(2) {
            circuit.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_8), &[q, q + 1]);
        }
    }

    circuit
}

fn mps_long_range_phase_circuit(n_qubits: usize, rounds: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }

    for _ in 0..rounds {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::Ry(0.17 + q as f64 * 0.003), &[q]);
        }
        for q in 0..(n_qubits / 2) {
            let partner = n_qubits - 1 - q;
            circuit.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_8), &[q, partner]);
        }
    }

    circuit
}

fn mps_measure_reset_circuit(n_qubits: usize, rounds: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, n_qubits);

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n_qubits.saturating_sub(1) {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }

    for round in 0..rounds {
        let parity = round % 2;
        for q in (parity..n_qubits).step_by(2) {
            circuit.add_measure(q, q);
            circuit.add_reset(q);
            circuit.add_gate(Gate::H, &[q]);
            if q + 1 < n_qubits {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }

    circuit
}

/// Linear cluster state on qubits `1..n` with qubit 0 as a re-attachable leaf.
/// Each round applies S to the first `sweep_qubits` qubits, re-entangles
/// qubit 0 via H plus CZ, then measures it. Z-measuring a leaf vertex removes
/// only that vertex, so the chain stays one entangled cluster and every
/// measurement pays the destabilizer rebuild. `sweep_qubits` sets the gate
/// load between measurements: 2 leaves the rebuild dominant, `n_qubits`
/// prices the shape under a full gate sweep.
fn interleaved_measure_chain_circuit(
    n_qubits: usize,
    rounds: usize,
    sweep_qubits: usize,
) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, rounds);

    for q in 1..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 1..n_qubits - 1 {
        circuit.add_gate(Gate::Cz, &[q, q + 1]);
    }

    for round in 0..rounds {
        for q in 0..sweep_qubits {
            circuit.add_gate(Gate::S, &[q]);
        }
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cz, &[0, 1]);
        circuit.add_measure(0, round);
    }

    circuit
}

fn compiled_filtered_bell_pairs_circuit(n_pairs: usize) -> Circuit {
    let mut circuit = circuits::independent_bell_pairs(n_pairs);
    let n = circuit.num_qubits;
    circuit.num_classical_bits = n;
    for i in 0..n {
        circuit.add_measure(i, i);
    }
    circuit
}

fn with_terminal_measurements(mut circuit: Circuit) -> Circuit {
    let n = circuit.num_qubits;
    circuit.num_classical_bits = n;
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    circuit
}

/// Thermal relaxation on every gate target. `t2 < 2 * t1` keeps the dephasing
/// rate above zero, so the event samples three branches rather than two.
fn thermal_noise_model(circuit: &Circuit) -> prism_q::NoiseModel {
    let mut model = prism_q::NoiseModel::uniform_depolarizing(circuit, 0.0);
    for (index, instruction) in circuit.instructions.iter().enumerate() {
        if let prism_q::Instruction::Gate { targets, .. } = instruction {
            model.after_gate[index] = targets
                .iter()
                .map(|&q| prism_q::NoiseEvent {
                    channel: prism_q::NoiseChannel::ThermalRelaxation {
                        t1: 100.0,
                        t2: 80.0,
                        gate_time: 1.0,
                    },
                    qubits: prism_q::circuit::SmallVec::from_slice(&[q]),
                })
                .collect();
        }
    }
    model
}

fn non_clifford_noise_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);
    for layer in 0..depth {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
            circuit.add_gate(Gate::T, &[q]);
            circuit.add_gate(Gate::Rz(0.03 * (layer + q + 1) as f64), &[q]);
        }
        for q in 0..n_qubits.saturating_sub(1) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    with_terminal_measurements(circuit)
}

fn run_mps_apply_only(circuit: &Circuit, max_bond_dim: usize) {
    let mut backend = MpsBackend::new(SEED, max_bond_dim);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    black_box(backend.classical_results());
}

// ---- Statevector: qubit-count sweeps ----

fn bench_statevector_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/random_d10");
    configure_group(&mut group);

    let mut sizes = vec![4, 8, 12, 16, 20];
    sizes.extend(high_qubit_sizes());
    for &n in &sizes {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qft_like");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16] {
        let circuit = qft_like_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

/// Setup cost of a variational sweep: 100 points of the same ansatz.
///
/// `rebuild` constructs and fuses the circuit at every point, which is what a
/// sweep does without a parameter surface. `bind` holds one [`PreparedCircuit`]
/// and rebinds, reusing the recorded fusion plan. Both arms live in this one
/// binary so an A/B compares them without a second build; see `benches/README.md`
/// on why a reference worktree cannot resolve a change that adds linked code.
///
/// Setup only, no simulation: the claim is amortized setup, and the apply cost
/// that would otherwise dominate is identical on both arms.
fn bench_parameter_sweep(c: &mut Criterion) {
    const POINTS: usize = 100;

    let mut group = c.benchmark_group("sweep_setup");
    configure_group(&mut group);

    // Quantum volume carries a far higher guard-to-site ratio than the ansatz
    // shapes, so it is the row that shows what replay costs rather than what it
    // saves.
    let shapes: [(&str, usize); 8] = [
        ("hea", 12),
        ("hea", 16),
        ("hea", 20),
        ("qaoa", 12),
        ("qaoa", 16),
        ("qaoa", 20),
        ("qv", 12),
        ("qv", 20),
    ];

    for (tag, n) in shapes {
        let build = move || match tag {
            "qaoa" => circuits::qaoa_circuit(n, 3, SEED),
            "qv" => circuits::quantum_volume_circuit(n, 5, SEED),
            _ => circuits::hardware_efficient_ansatz(n, 5, SEED),
        };
        let template = build();
        let params = Parameters::all_rotations(&template);
        let points: Vec<Vec<f64>> = (0..POINTS)
            .map(|k| {
                let mut rng = ChaCha8Rng::seed_from_u64(SEED.wrapping_add(k as u64));
                (0..params.num_slots())
                    .map(|_| rng.random::<f64>() * std::f64::consts::TAU)
                    .collect()
            })
            .collect();

        group.bench_function(BenchmarkId::new("rebuild", format!("{tag}/{n}")), |b| {
            b.iter(|| {
                for values in &points {
                    let mut circuit = build();
                    for link in params.links() {
                        if let Instruction::Gate {
                            gate:
                                Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::Rzz(t) | Gate::P(t),
                            ..
                        } = &mut circuit.instructions[link.instruction]
                        {
                            *t = values[link.slot];
                        }
                    }
                    black_box(fuse_circuit(&circuit, true).into_owned());
                }
            });
        });

        group.bench_function(BenchmarkId::new("bind", format!("{tag}/{n}")), |b| {
            let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
            b.iter(|| {
                for values in &points {
                    black_box(prepared.bind_fused(values).unwrap().instructions.len());
                }
            });
        });
    }

    group.finish();
}

fn bench_statevector_hea(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/hea_l5");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_clifford(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/clifford_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_qft_textbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qft_textbook");
    configure_group(&mut group);

    let mut sizes = vec![4, 8, 12, 16, 20, 22, 24, 26];
    sizes.extend(high_qubit_sizes());
    for &n in &sizes {
        let circuit = circuits::qft_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_qpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qpe_t_gate");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20, 22] {
        let circuit = circuits::phase_estimation_circuit(n);
        let label = format!("{}q", n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Statevector: QAOA ----

fn bench_statevector_qaoa(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qaoa_l3");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::qaoa_circuit(n, 3, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// One first-order Trotter step of the seeded Jordan-Wigner two-body operator,
// truncated to the 200 largest coefficients. The recognizing constructor
// lowers weight-1 and ZZ terms, so the native arm mixes named rotations with
// `PauliRot` strings; the ladder arm pre-expands those strings, so the delta
// isolates the native kernel against the fused CNOT-ladder form.
fn trotter_step_circuit(n: usize) -> Circuit {
    let terms = circuits::jordan_wigner_hamiltonian(n, 200, SEED);
    let dt = 0.05;
    let mut c = Circuit::new(n, 0);
    for (coefficient, factors) in &terms {
        if factors.is_empty() {
            continue;
        }
        c.add_pauli_rotation(2.0 * coefficient * dt, factors);
    }
    c
}

fn bench_statevector_trotter(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/trotter");
    configure_group(&mut group);

    for &n in &[16, 20] {
        let native = trotter_step_circuit(n);
        let ladder = prism_q::circuit::expand_pauli_rotations(&native).into_owned();
        group.bench_with_input(BenchmarkId::new("native", n), &native, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
        group.bench_with_input(BenchmarkId::new("ladder", n), &ladder, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// Gates on the parallel `DiagonalBatch` sweep: the fused stream carrying a
// `DiagonalBatch` is pinned by `fusion_diag_mixed_batches_and_matches_unfused`.
fn bench_statevector_diag_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/diag_mixed_l6");
    configure_group(&mut group);

    for &n in &[16, 20] {
        let circuit = circuits::diagonal_mixed_circuit(n, 6, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_qv(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/qv");
    configure_group(&mut group);

    for &n in &[8, 12, 16, 20] {
        let circuit = circuits::quantum_volume_circuit(n, n, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_statevector_w_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/w_state");
    configure_group(&mut group);

    for &n in &[8, 12, 16, 20] {
        let circuit = circuits::w_state_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Statevector: depth sweep ----

fn bench_statevector_depth_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/depth_sweep_12q");
    configure_group(&mut group);

    let depths = vec![5, 10, 20, 50, 100];
    for &depth in &depths {
        let circuit = circuits::random_circuit(12, depth, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(depth), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Statevector: entanglement structure ----

fn bench_statevector_entanglement(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/entanglement_16q_d10");
    configure_group(&mut group);

    let sparse = sparse_entanglement_circuit(16, 10);
    group.bench_function("sparse", |b| {
        b.iter(|| {
            run_with(BackendKind::Statevector, &sparse, 42).unwrap();
        });
    });

    let dense = dense_entanglement_circuit(16, 10);
    group.bench_function("dense", |b| {
        b.iter(|| {
            run_with(BackendKind::Statevector, &dense, 42).unwrap();
        });
    });

    group.finish();
}

// ---- Statevector: scalability sweep (2–26 qubits) ----

fn bench_statevector_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("statevector/scalability_d5");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    for n in (2..=26).step_by(2) {
        let circuit = circuits::random_circuit(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Statevector, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Stabilizer backend ----

fn bench_stabilizer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer/scaling");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500, 1000, 5000] {
        let circuit = random_clifford_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Stabilizer, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_stabilizer_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer/measurement");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500, 1000, 5000] {
        let mut circuit = Circuit::new(n, n);
        circuit.add_gate(Gate::H, &[0]);
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        group.bench_with_input(
            BenchmarkId::new("ghz_measure_all", n),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::Stabilizer, circ, 42).unwrap();
                });
            },
        );
    }
    group.finish();
}

// ---- Factored stabilizer backend ----

fn bench_factored_stabilizer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored_stabilizer/scaling");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500, 1000] {
        let circuit = random_clifford_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_factored_stabilizer_local(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored_stabilizer/local_blocks");
    configure_group(&mut group);

    for &(blocks, block_size) in &[(5, 2), (5, 4), (10, 5), (20, 5), (50, 5)] {
        let n = blocks * block_size;
        let circuit = circuits::local_clifford_blocks(blocks, block_size, 10, SEED);
        group.bench_with_input(
            BenchmarkId::new(format!("{}x{}", blocks, block_size), n),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_factored_stabilizer_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored_stabilizer/measurement");
    configure_group(&mut group);

    for &n in &[10, 50, 100, 500] {
        let mut circuit = Circuit::new(n, n);
        circuit.add_gate(Gate::H, &[0]);
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        group.bench_with_input(
            BenchmarkId::new("ghz_measure_all", n),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
                });
            },
        );
    }

    for &n in &[50, 100, 500] {
        let circuit = interleaved_measure_chain_circuit(n, 10, n);
        group.bench_with_input(
            BenchmarkId::new("interleaved_chain_r10", n),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
                });
            },
        );

        let lean = interleaved_measure_chain_circuit(n, 10, 2);
        group.bench_with_input(
            BenchmarkId::new("interleaved_chain_lean_r10", n),
            &lean,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::FactoredStabilizer, circ, 42).unwrap();
                });
            },
        );
    }
    group.finish();
}

// ---- Sparse backend ----

fn bench_sparse_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse/random_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Sparse, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_sparse_low_entanglement(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse/low_entanglement");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = sparse_entanglement_circuit(n, 5);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Sparse, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

// ---- MPS backend ----

fn bench_mps_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps/random_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Mps { max_bond_dim: 64 }, circ, 42).unwrap();
            });
        });
    }

    // Bond 256 is the Auto-dispatch and Python default. This family peaks at
    // bond 4, far below either cap, so the row does the same work as the 64-cap
    // row and guards against the cap entering the cost path.
    let circuit = circuits::random_circuit(20, 10, SEED);
    group.bench_with_input(BenchmarkId::new("b256", 20), &circuit, |b, circ| {
        b.iter(|| {
            run_with(BackendKind::Mps { max_bond_dim: 256 }, circ, 42).unwrap();
        });
    });
    group.finish();
}

fn bench_mps_linear_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps/linear_chain_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 16, 32, 64] {
        let circuit = dense_entanglement_circuit(n, 10);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Mps { max_bond_dim: 64 }, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_mps_hotspots(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps/hotspots");
    configure_group(&mut group);

    for &n in &[16, 32] {
        let adjacent = mps_adjacent_phase_circuit(n, 4);
        group.bench_with_input(
            BenchmarkId::new("adjacent_cp_r4", n),
            &adjacent,
            |b, circ| {
                b.iter(|| run_mps_apply_only(circ, 64));
            },
        );

        let long_range = mps_long_range_phase_circuit(n, 4);
        group.bench_with_input(
            BenchmarkId::new("long_range_cp_r4", n),
            &long_range,
            |b, circ| {
                b.iter(|| run_mps_apply_only(circ, 64));
            },
        );
    }

    let meas_reset = mps_measure_reset_circuit(32, 3);
    group.bench_function("measure_reset_32q_r3", |b| {
        b.iter(|| run_mps_apply_only(&meas_reset, 64));
    });

    // Routed long-range pairs sit adjacent after the SWAPs and peak at bond 2,
    // so this row matches the 64-cap row and guards the routing path against
    // cap-dependent cost at the bond 256 default.
    let long_range_256 = mps_long_range_phase_circuit(32, 4);
    group.bench_with_input(
        BenchmarkId::new("long_range_cp_r4_b256", 32),
        &long_range_256,
        |b, circ| {
            b.iter(|| run_mps_apply_only(circ, 256));
        },
    );

    group.finish();
}

// ---- Native shot sampling on the polynomial-state backends ----

/// Terminal measurements on every qubit, the shape the native samplers serve.
fn measure_all(circuit: &Circuit) -> Circuit {
    let mut measured = Circuit::new(circuit.num_qubits, circuit.num_qubits);
    measured.instructions = circuit.instructions.clone();
    for q in 0..circuit.num_qubits {
        measured.add_measure(q, q);
    }
    measured
}

/// Sizes past the dense probability cap, where the sampler is the only path to
/// a bitstring. The 24q row stays inside the cap so the polynomial cost can be
/// read against the dense route on the same circuit family.
const SAMPLING_QUBITS: [usize; 4] = [24, 32, 48, 64];

const SAMPLING_SHOTS: usize = 1_000;

fn bench_mps_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps/sampling");
    configure_group(&mut group);

    for &n in &SAMPLING_QUBITS {
        let circuit = measure_all(&dense_entanglement_circuit(n, 4));
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                black_box(
                    run_shots_with(
                        BackendKind::Mps { max_bond_dim: 32 },
                        circ,
                        SAMPLING_SHOTS,
                        SEED,
                    )
                    .unwrap(),
                )
            });
        });
    }

    group.finish();
}

fn bench_sparse_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse/sampling");
    configure_group(&mut group);

    for &n in &SAMPLING_QUBITS {
        let circuit = measure_all(&sparse_entanglement_circuit(n, 2));
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                black_box(run_shots_with(BackendKind::Sparse, circ, SAMPLING_SHOTS, SEED).unwrap())
            });
        });
    }
    group.finish();
}

// ---- Product state backend ----

fn bench_product_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("product/scaling_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 16, 32, 64, 128, 256] {
        let circuit = random_single_qubit_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::ProductState, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

/// Product sampling is `O(shots·n)` with no `2^n` term anywhere, so the widths
/// go past where the other samplers stop. The row exists to show the cost is
/// linear in `n`: each step doubles the width and should double the time.
const PRODUCT_SAMPLING_QUBITS: [usize; 4] = [64, 128, 256, 512];

fn bench_product_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("product/sampling");
    configure_group(&mut group);

    for &n in &PRODUCT_SAMPLING_QUBITS {
        let circuit = measure_all(&random_single_qubit_circuit(n, 10, SEED));
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                black_box(
                    run_shots_with(BackendKind::ProductState, circ, SAMPLING_SHOTS, SEED).unwrap(),
                )
            });
        });
    }
    group.finish();
}

// ---- Tensor Network backend ----

fn bench_tn_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/random_d10");
    configure_group(&mut group);

    for &n in &[4, 8, 12] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::TensorNetwork, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_tn_linear_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/linear_chain");
    configure_group(&mut group);

    for &n in &[4, 8, 12] {
        let circuit = dense_entanglement_circuit(n, 5);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::TensorNetwork, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "bench-internal"))]
fn bench_tn_scalar_expectation(_c: &mut Criterion) {}

/// Wide low-treewidth rows. The other two `tn/` groups run the dense terminal,
/// whose intermediates are `2^n` whatever the circuit structure; here the
/// largest holds 128 to 256 elements, flat across 20 to 50 qubits.
#[cfg(feature = "bench-internal")]
fn bench_tn_scalar_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/scalar_hea_l2");
    configure_group(&mut group);

    for &n in &[20, 30, 40, 50] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
        let observable = [PauliTerm::z(0), PauliTerm::z(n / 2)];
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                black_box(scalar_expectation(circ, &observable).unwrap());
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "bench-internal"))]
fn bench_tn_scalar_depth(_c: &mut Criterion) {}

/// Rising-treewidth rows: 20 qubits, layer count swept, so the intermediates
/// grow instead of staying flat as they do in `tn/scalar_hea_l2`. 4 layers is
/// where the largest first clears `MIN_PAR_ELEMS`, below which the parallel arms
/// of `contract` and `transpose` never run.
#[cfg(feature = "bench-internal")]
fn bench_tn_scalar_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/scalar_depth_20q");
    configure_group(&mut group);

    let observable = [PauliTerm::z(0), PauliTerm::z(10)];
    for &layers in &[4, 5, 6, 7] {
        let circuit = circuits::hardware_efficient_ansatz(20, layers, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(layers), &circuit, |b, circ| {
            b.iter(|| {
                black_box(scalar_expectation(circ, &observable).unwrap());
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "bench-internal"))]
fn bench_tn_scalar_wide_deep(_c: &mut Criterion) {}

/// Wide and deep together, the only rows where width drives the faer arm.
///
/// `tn/scalar_hea_l2` sweeps the same widths two layers deep, where the largest
/// intermediate holds 256 elements and no contraction reaches
/// `MIN_FAER_GEMM_WORK`; `tn/scalar_depth_20q` reaches it but only at 20 qubits.
/// Six layers puts 12 contractions over the threshold at 20 qubits and 37 at 50,
/// so a change to the crossover shows up here as a function of width. Seven
/// layers lives in `tn/scalar_hea_l7`, where tree quality is the variable.
#[cfg(feature = "bench-internal")]
fn bench_tn_scalar_wide_deep(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/scalar_hea_l6");
    configure_group(&mut group);

    for &n in &[20, 30, 40, 50] {
        let circuit = circuits::hardware_efficient_ansatz(n, 6, SEED);
        let observable = [PauliTerm::z(0), PauliTerm::z(n / 2)];
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                black_box(scalar_expectation(circ, &observable).unwrap());
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "bench-internal"))]
fn bench_tn_scalar_tree_quality(_c: &mut Criterion) {}

/// The rows where contraction tree quality is the variable.
///
/// Seven layers is where the greedy planner's tree is on record as
/// non-monotonic in width: peak intermediate 16.8M elements at 30 and 50
/// qubits against 1.05M at 40. A planner change moves these rows through the
/// tree it picks, not through kernel arithmetic, which the depth and width
/// sweeps above already cover.
#[cfg(feature = "bench-internal")]
fn bench_tn_scalar_tree_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/scalar_hea_l7");
    configure_group(&mut group);

    for &n in &[30, 40, 50] {
        let circuit = circuits::hardware_efficient_ansatz(n, 7, SEED);
        let observable = [PauliTerm::z(0), PauliTerm::z(n / 2)];
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                black_box(scalar_expectation(circ, &observable).unwrap());
            });
        });
    }
    group.finish();
}

/// Reduced density matrix on a long chain, past the dense query ceiling.
///
/// The query doubles the network against its conjugate and contracts with one
/// ket and one bra leg open, so the rows exercise tree choice on a doubled
/// low-treewidth network at widths where `probabilities()` cannot answer.
/// Depth 4 peaks at 64 elements whatever the width, so those rows weigh
/// per-contraction overhead; depth 8 at 60 qubits peaks at 1M elements and
/// weighs the arithmetic.
fn bench_tn_rdm_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/rdm_chain");
    configure_group(&mut group);

    for &(n, depth) in &[(40usize, 4usize), (60, 4), (60, 8)] {
        let circuit = circuits::cz_chain_circuit(n, depth, SEED);
        let mut tn = TensorNetworkBackend::new(SEED);
        tn.init(n, 0).unwrap();
        for inst in &circuit.instructions {
            tn.apply(inst).unwrap();
        }
        let label = format!("{n}_d{depth}");
        group.bench_with_input(BenchmarkId::from_parameter(label), &tn, |b, backend| {
            b.iter(|| {
                black_box(backend.reduced_density_matrix_1q(n / 2).unwrap());
            });
        });
    }
    group.finish();
}

/// Build a CZ-chain circuit with one measurement and one reset at half depth.
fn mid_measured_chain(n: usize, depth: usize) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let singles = [Gate::H, Gate::S, Gate::T, Gate::X];
    let mut c = Circuit::new(n, 1);
    for layer in 0..depth {
        for q in 0..n {
            c.add_gate(singles[rng.random_range(0..singles.len())].clone(), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n - 1).step_by(2) {
            c.add_gate(Gate::Cz, &[q, q + 1]);
        }
        if layer + 1 == depth / 2 {
            c.add_measure(n / 2, 0);
            c.add_reset(n / 4);
        }
    }
    c
}

/// Mid-circuit measurement on a chain, where the measurement path's cost
/// lands mid-run rather than at the terminal readout.
fn bench_tn_midmeasure_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/midmeasure_chain");
    configure_group(&mut group);

    for &n in &[16, 20] {
        let circuit = mid_measured_chain(n, 4);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::TensorNetwork, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

/// Depolarizing trajectories on the chain shape, the priced noisy row.
///
/// Terminal measurements dominate the per-shot cost, so this row moves with
/// the measurement path as much as with the noise machinery.
fn bench_tn_noisy_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("tn/noisy_chain");
    configure_group(&mut group);

    let n = 16;
    let mut circuit = circuits::cz_chain_circuit(n, 4, SEED);
    circuit.measure_all();
    let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
        b.iter(|| {
            run_shots_with_noise(BackendKind::TensorNetwork, circ, &noise, 100, 42).unwrap();
        });
    });
    group.finish();
}

// ---- Cross-backend comparisons ----

fn bench_compare_clifford(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/clifford_d10");
    configure_group(&mut group);

    let backends: &[(&str, BackendKind)] = &[
        ("statevector", BackendKind::Statevector),
        ("stabilizer", BackendKind::Stabilizer),
        ("sparse", BackendKind::Sparse),
        ("mps_64", BackendKind::Mps { max_bond_dim: 64 }),
        ("auto", BackendKind::Auto),
    ];

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = random_clifford_circuit(n, 10, SEED);
        for &(name, ref kind) in backends {
            group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                b.iter(|| {
                    run_with(kind.clone(), circ, 42).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_compare_single_qubit(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/single_qubit_d10");
    configure_group(&mut group);

    let backends: &[(&str, BackendKind)] = &[
        ("statevector", BackendKind::Statevector),
        ("product", BackendKind::ProductState),
        ("auto", BackendKind::Auto),
    ];

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = random_single_qubit_circuit(n, 10, SEED);
        for &(name, ref kind) in backends {
            group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                b.iter(|| {
                    run_with(kind.clone(), circ, 42).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_compare_general(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare/general_d10");
    configure_group(&mut group);

    let backends: &[(&str, BackendKind)] = &[
        ("statevector", BackendKind::Statevector),
        ("sparse", BackendKind::Sparse),
        ("mps_64", BackendKind::Mps { max_bond_dim: 64 }),
        ("auto", BackendKind::Auto),
    ];

    for &n in &[4, 8, 12, 16, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        for &(name, ref kind) in backends {
            group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                b.iter(|| {
                    run_with(kind.clone(), circ, 42).unwrap();
                });
            });
        }
    }
    group.finish();
}

// ---- Auto dispatch sweeps ----

fn bench_auto_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/random_d10");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_qft(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/qft_like");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16] {
        let circuit = qft_like_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_qft_textbook(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/qft_textbook");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20, 22] {
        let circuit = circuits::qft_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_qpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/qpe_t_gate");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20, 22] {
        let circuit = circuits::phase_estimation_circuit(n);
        let label = format!("{}q", n);
        group.bench_with_input(BenchmarkId::from_parameter(label), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_hea(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/hea_l5");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_clifford(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/clifford_d10");
    configure_group(&mut group);

    for &n in &[4, 6, 8, 10, 12, 14, 16, 18, 20] {
        let circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_auto_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto/scalability_d5");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    for n in (2..=26).step_by(2) {
        let circuit = circuits::random_circuit(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_with(BackendKind::Auto, circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Decomposition benchmarks ----

fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition");
    configure_group(&mut group);

    for &total_q in &[8, 12, 16, 20] {
        let n_pairs = total_q / 2;
        let circuit = circuits::independent_bell_pairs(n_pairs);

        group.bench_with_input(
            BenchmarkId::new("bell_decomposed", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bell_monolithic", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    let mut sv = prism_q::StatevectorBackend::new(42);
                    sim::run_on(&mut sv, circ).unwrap();
                });
            },
        );
    }

    for &block_size in &[2, 4, 5, 10] {
        let num_blocks = 20 / block_size;
        let circuit = circuits::independent_random_blocks(num_blocks, block_size, 5, SEED);

        group.bench_with_input(
            BenchmarkId::new(format!("20q_block{}_decomposed", block_size), block_size),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_with(BackendKind::Statevector, circ, 42).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("20q_block{}_monolithic", block_size), block_size),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    let mut sv = prism_q::StatevectorBackend::new(42);
                    sim::run_on(&mut sv, circ).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ---- Adjoint gradient benchmarks ----

/// A small weighted Z-chain Hamiltonian over the first few qubits.
fn z_chain_hamiltonian(n: usize) -> Vec<(f64, Vec<PauliTerm>)> {
    (0..n - 1)
        .map(|q| (1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]))
        .collect()
}

/// Add `delta` to the angle of every gate bound to parameter `slot`.
fn shift_slot(circuit: &Circuit, params: &Parameters, slot: usize, delta: f64) -> Circuit {
    let mut out = circuit.clone();
    for link in params.links().iter().filter(|l| l.slot == slot) {
        if let Instruction::Gate {
            gate: Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::Rzz(t) | Gate::P(t),
            ..
        } = &mut out.instructions[link.instruction]
        {
            *t += delta;
        }
    }
    out
}

/// Central finite-difference gradient over all parameters, the current user
/// alternative to the adjoint method (two forward expectation runs per slot).
fn finite_diff_gradient(
    circuit: &Circuit,
    hamiltonian: &[(f64, Vec<PauliTerm>)],
    params: &Parameters,
) -> Vec<f64> {
    let observables: Vec<Vec<PauliTerm>> = hamiltonian.iter().map(|(_, p)| p.clone()).collect();
    let eps = 1e-5;
    let expval = |c: &Circuit| -> f64 {
        let per_term = run_expectation_values(c, &observables, 42).unwrap();
        hamiltonian
            .iter()
            .zip(per_term)
            .map(|((coeff, _), v)| coeff * v)
            .sum()
    };
    (0..params.num_slots())
        .map(|slot| {
            let plus = expval(&shift_slot(circuit, params, slot, eps));
            let minus = expval(&shift_slot(circuit, params, slot, -eps));
            (plus - minus) / (2.0 * eps)
        })
        .collect()
}

fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient/hea_l2");
    configure_group(&mut group);

    for &n in &[14, 18, 20, 22] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
        let params = Parameters::all_rotations(&circuit);
        let ham = z_chain_hamiltonian(n);

        group.bench_with_input(BenchmarkId::new("adjoint", n), &circuit, |b, circ| {
            b.iter(|| black_box(run_expectation_gradient(circ, &ham, &params, 42).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("finitediff", n), &circuit, |b, circ| {
            b.iter(|| black_box(finite_diff_gradient(circ, &ham, &params)));
        });
        group.bench_with_input(BenchmarkId::new("paramshift", n), &circuit, |b, circ| {
            b.iter(|| black_box(run_expectation_gradient_shift(circ, &ham, &params, 42).unwrap()));
        });
    }

    group.finish();
}

fn bench_gradient_qaoa(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient/qaoa_l1");
    configure_group(&mut group);

    for &n in &[14, 18, 20] {
        let circuit = circuits::qaoa_circuit(n, 1, SEED);
        let params = Parameters::all_rotations(&circuit);
        let ham = z_chain_hamiltonian(n);

        group.bench_with_input(BenchmarkId::new("adjoint", n), &circuit, |b, circ| {
            b.iter(|| black_box(run_expectation_gradient(circ, &ham, &params, 42).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("finitediff", n), &circuit, |b, circ| {
            b.iter(|| black_box(finite_diff_gradient(circ, &ham, &params)));
        });
    }

    group.finish();
}

/// A fixed entangling prefix (non-trainable) followed by a trainable
/// single-qubit-rotation layer, with a local single-qubit observable. Exercises
/// the backward-sweep early termination (prefix skipped) and light-cone
/// sandwich pruning (only the in-cone trainable gate contributes).
fn prefix_local_circuit(n: usize, prefix_layers: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for layer in 0..prefix_layers {
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        for q in (layer % 2..n - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    for q in 0..n {
        c.add_gate(Gate::Ry(0.1 + 0.01 * q as f64), &[q]);
    }
    c
}

fn bench_gradient_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient/prefix_local");
    configure_group(&mut group);

    for &n in &[16, 18, 20] {
        let circuit = prefix_local_circuit(n, 6);
        let params = Parameters::all_rotations(&circuit);
        let ham = vec![(1.0, vec![PauliTerm::z(0)])];

        group.bench_with_input(BenchmarkId::new("adjoint", n), &circuit, |b, circ| {
            b.iter(|| black_box(run_expectation_gradient(circ, &ham, &params, 42).unwrap()));
        });
    }

    group.finish();
}

// A Trotter ansatz over the largest Jordan-Wigner strings of the seeded
// two-body operator, on an alternating occupation reference. The recognizing
// constructor lowers the weight-1 and ZZ generators, so the stream mixes named
// rotations with `PauliRot` and the gradient runs over both.
fn vqe_ansatz(n: usize, generators: &[Vec<PauliTerm>], values: &[f64]) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        c.add_gate(Gate::X, &[q]);
    }
    for (theta, factors) in values.iter().zip(generators) {
        c.add_pauli_rotation(*theta, factors);
    }
    c
}

/// One variational loop iteration: evaluate the weighted observable, take the
/// adjoint gradient, step the parameters.
///
/// `rebuild` constructs the ansatz at every iteration, which is what a loop
/// without a parameter surface does; `bind` holds one [`PreparedCircuit`] and
/// rebinds. Both arms live in this one binary so an A/B compares them without a
/// second build; see `benches/README.md` on why a reference worktree cannot
/// resolve a change that adds linked code.
///
/// The `sweep_setup` group times the same two arms with no simulation. This one
/// pays the evaluation and gradient cost the setup is amortized over, so the two
/// rows together say what share of a real iteration the parameter surface moves.
fn bench_vqe_loop(c: &mut Criterion) {
    const GENERATORS: usize = 16;
    const OBSERVABLE_TERMS: usize = 64;
    const LEARNING_RATE: f64 = 0.01;

    let mut group = c.benchmark_group("vqe");
    configure_group(&mut group);

    for &n in &[12, 16] {
        let terms = circuits::jordan_wigner_hamiltonian(n, OBSERVABLE_TERMS, SEED);
        let observable = PauliObservable::from_terms(terms.clone()).unwrap();
        let generators: Vec<Vec<PauliTerm>> = terms
            .iter()
            .map(|(_, factors)| factors.clone())
            .filter(|factors| !factors.is_empty())
            .take(GENERATORS)
            .collect();
        let values: Vec<f64> = (0..generators.len())
            .map(|k| 0.05 + 0.01 * k as f64)
            .collect();

        let template = vqe_ansatz(n, &generators, &values);
        let params = Parameters::all_rotations(&template);

        let iteration = |circuit: &Circuit| {
            let energy = run_observable_expectation(circuit, &observable, 42).unwrap();
            let grad = run_expectation_gradient(circuit, &terms, &params, 42).unwrap();
            let stepped: Vec<f64> = values
                .iter()
                .zip(&grad.gradient)
                .map(|(v, g)| v - LEARNING_RATE * g)
                .collect();
            (energy.mean, stepped)
        };

        group.bench_function(BenchmarkId::new("rebuild", n), |b| {
            b.iter(|| {
                let circuit = vqe_ansatz(n, &generators, &values);
                black_box(iteration(&circuit));
            });
        });

        group.bench_function(BenchmarkId::new("bind", n), |b| {
            let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
            b.iter(|| {
                let circuit = prepared.bind(&values).unwrap();
                black_box(iteration(circuit));
            });
        });
    }

    group.finish();
}

fn bench_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expectation/pauli_sum");
    configure_group(&mut group);

    for &n in &[14, 16, 18, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
        let observables: Vec<Vec<PauliTerm>> = (0..n - 1)
            .map(|q| vec![PauliTerm::z(q), PauliTerm::z(q + 1)])
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| black_box(run_expectation_values(circ, &observables, 42).unwrap()));
        });
    }

    group.finish();
}

/// `expectation/pauli_sum` runs `Auto` and resolves to the statevector family,
/// so it never reaches the factored observable path. `hardware_efficient_ansatz`
/// is one connected component, so the factored state is a single sub-state
/// spanning every qubit.
fn bench_expectation_factored(c: &mut Criterion) {
    let mut group = c.benchmark_group("expectation/pauli_sum_factored");
    configure_group(&mut group);

    for &n in &[16, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
        let observables: Vec<Vec<PauliTerm>> = (0..n - 1)
            .map(|q| vec![PauliTerm::z(q), PauliTerm::z(q + 1)])
            .collect();

        for &(name, ref kind) in &[
            ("factored", BackendKind::Factored),
            ("statevector", BackendKind::Statevector),
        ] {
            group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                b.iter(|| {
                    black_box(
                        sim::simulate(circ)
                            .backend(kind.clone())
                            .seed(42)
                            .expectation_values(&observables)
                            .unwrap(),
                    )
                });
            });
        }
    }

    group.finish();
}

/// Observable evaluation on the backends that answer from their own
/// representation instead of a dense amplitude buffer. Same circuit family and
/// observable shape as `expectation/pauli_sum_factored`, with the same
/// `statevector` control arm; sparse stops at 16 qubits because its state is
/// dense here and the per-observable walk is a hash lookup per stored
/// amplitude.
fn bench_expectation_native(c: &mut Criterion) {
    for &(group_name, ref kind, ref sizes) in &[
        (
            "expectation/pauli_sum_sparse",
            BackendKind::Sparse,
            vec![12usize, 16],
        ),
        (
            "expectation/pauli_sum_mps",
            BackendKind::Mps { max_bond_dim: 64 },
            vec![16usize, 20],
        ),
    ] {
        let mut group = c.benchmark_group(group_name);
        configure_group(&mut group);

        for &n in sizes {
            let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
            let observables: Vec<Vec<PauliTerm>> = (0..n - 1)
                .map(|q| vec![PauliTerm::z(q), PauliTerm::z(q + 1)])
                .collect();

            for &(name, ref arm) in &[
                ("native", kind.clone()),
                ("statevector", BackendKind::Statevector),
            ] {
                group.bench_with_input(BenchmarkId::new(name, n), &circuit, |b, circ| {
                    b.iter(|| {
                        black_box(
                            sim::simulate(circ)
                                .backend(arm.clone())
                                .seed(42)
                                .expectation_values(&observables)
                                .unwrap(),
                        )
                    });
                });
            }
        }

        group.finish();
    }
}

/// Observable evaluation on a prepared density matrix.
///
/// Evolution is outside the timed section here, unlike the groups above. A
/// `4^n` buffer costs seconds to evolve and microseconds to read observables
/// off, so a whole-pipeline row cannot resolve a change to the reduction.
/// No in-group control arm: the statevector answers observables through the
/// simulation helper rather than the trait method, so it cannot be timed on
/// the same footing. The statevector arms of the groups above serve instead.
fn bench_expectation_density_matrix(c: &mut Criterion) {
    use prism_q::backend::Backend;
    use prism_q::backend::density_matrix::DensityMatrixBackend;

    let mut group = c.benchmark_group("expectation/pauli_sum_density_matrix");
    configure_group(&mut group);

    for &n in &[8usize, 10] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
        let observables: Vec<Vec<PauliTerm>> = (0..n - 1)
            .map(|q| vec![PauliTerm::z(q), PauliTerm::z(q + 1)])
            .collect();

        let mut dm = DensityMatrixBackend::new(SEED);
        sim::run_on(&mut dm, &circuit).unwrap();
        group.bench_with_input(BenchmarkId::new("native", n), &dm, |b, backend| {
            b.iter(|| black_box(backend.pauli_expectations(&observables).unwrap()));
        });
    }

    group.finish();
}

/// Weighted-observable evaluation at a molecular-Hamiltonian term count:
/// 2000 Jordan-Wigner strings per size. The `grouped` arm is the
/// qubit-wise-commuting route with variance; the `ungrouped` control is the
/// pre-existing batched per-term path with weights applied caller-side, which
/// the grouped route shares no evaluation code with. Observable evaluation
/// dominates the iteration, not the circuit run.
fn bench_expectation_grouped(c: &mut Criterion) {
    let mut group = c.benchmark_group("expectation/pauli_sum_grouped");
    group.sample_size(10);
    if common::is_fast() {
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
    } else {
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(10));
    }

    for &n in &[16, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, SEED);
        let terms = circuits::jordan_wigner_hamiltonian(n, 2000, SEED);
        let observable = PauliObservable::from_terms(terms.clone()).unwrap();
        let obs_vecs: Vec<Vec<PauliTerm>> =
            terms.iter().map(|(_, factors)| factors.clone()).collect();
        let coefficients: Vec<f64> = terms.iter().map(|(c, _)| *c).collect();

        group.bench_with_input(BenchmarkId::new("grouped", n), &circuit, |b, circ| {
            b.iter(|| black_box(run_observable_expectation(circ, &observable, 42).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("ungrouped", n), &circuit, |b, circ| {
            b.iter(|| {
                let values = run_expectation_values(circ, &obs_vecs, 42).unwrap();
                let mean: f64 = coefficients.iter().zip(&values).map(|(c, v)| c * v).sum();
                black_box(mean)
            });
        });
    }

    group.finish();
}

// ---- Factored backend benchmarks ----

fn bench_factored_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/random_d10");
    configure_group(&mut group);

    for &n in &[16, 20, 24] {
        let circuit = circuits::random_circuit(n, 10, SEED);

        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Statevector, circ, 42).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Factored, circ, 42).unwrap());
        });
    }

    group.finish();
}

fn bench_factored_independent(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/independent");
    configure_group(&mut group);

    for &total_q in &[16, 20, 24] {
        let circuit = circuits::independent_random_blocks(total_q / 4, 4, 5, SEED);

        group.bench_with_input(
            BenchmarkId::new("statevector", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| run_with(BackendKind::Statevector, circ, 42).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("factored", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| run_with(BackendKind::Factored, circ, 42).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_factored_sim_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/sim_only_d10");
    configure_group(&mut group);

    for &n in &[16, 20, 24] {
        let circuit = circuits::random_circuit(n, 10, SEED);

        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Statevector, circ, 42).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Factored, circ, 42).unwrap());
        });
    }

    group.finish();
}

fn bench_factored_dynamic(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/dynamic_advantage");
    configure_group(&mut group);

    // Independent blocks with ONE bridging CX at the end. The bridge merges the
    // first and last block only, so static analysis still finds several
    // components and the run takes the decomposed route rather than the
    // factored backend's own multi-block terminal.
    for &total_q in &[16, 20, 24] {
        let block_size = 4;
        let num_blocks = total_q / block_size;
        let mut circuit = circuits::independent_random_blocks(num_blocks, block_size, 5, SEED);
        // Add one CX bridging first and last block (prevents static decomposition)
        circuit.add_gate(Gate::Cx, &[0, total_q - 1]);

        group.bench_with_input(
            BenchmarkId::new("statevector", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| run_with(BackendKind::Statevector, circ, 42).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("factored", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| run_with(BackendKind::Factored, circ, 42).unwrap());
            },
        );
    }

    group.finish();
}

/// The factored backend's multi-block probability terminal, which
/// `dynamic_advantage` does not reach: that group decomposes at the sim layer,
/// which returns per-block probabilities without merging. Here the largest
/// component is `n - 2`, so `should_decompose` declines and the whole circuit
/// runs on one backend while the factored state stays in two sub-states. The
/// statevector row is the reference for the same terminal cost.
fn bench_factored_partial_independence(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/partial_independence");
    configure_group(&mut group);

    for &n in &[16, 20] {
        let circuit = circuits::partially_independent_circuit(n, 5, SEED);
        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Statevector, circ, 42).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Factored, circ, 42).unwrap());
        });
    }

    group.finish();
}

/// Non-Pauli trajectory noise on the factored backend, the only path that reaches
/// `Backend::apply_1q_matrix`. Pauli noise routes to gate application instead, so
/// `noisy_sampling` does not cover this at all.
///
/// Four-qubit blocks keep each sub-state at 16 amplitudes, so per-call cost in
/// that method is a visible fraction of the kernel. Deep and shot-light because a
/// shot allocates a fresh backend per block: at depth 5 with 256 shots that
/// construction read a 36% spread on identical code.
///
/// The thermal rows measure the same method with a three-branch sampler instead
/// of two, since thermal relaxation is amplitude damping composed with pure
/// dephasing and the composition folds to one probability read and one matrix
/// pass. The group has no in-group control; gate a change here against its own
/// control column.
fn bench_factored_noise_kraus(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/noise_kraus");
    configure_group(&mut group);

    for &total_q in &[16, 24] {
        let circuit = with_terminal_measurements(circuits::independent_random_blocks(
            total_q / 4,
            4,
            60,
            SEED,
        ));

        let damping = prism_q::NoiseModel::with_amplitude_damping(&circuit, 0.01);
        group.bench_with_input(
            BenchmarkId::new("amplitude_damping", total_q),
            &circuit,
            |b, circ| {
                b.iter(|| {
                    run_shots_with_noise(BackendKind::Factored, circ, &damping, 32, SEED).unwrap();
                });
            },
        );

        let thermal = thermal_noise_model(&circuit);
        group.bench_with_input(BenchmarkId::new("thermal", total_q), &circuit, |b, circ| {
            b.iter(|| {
                run_shots_with_noise(BackendKind::Factored, circ, &thermal, 32, SEED).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_factored_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("factored/dense");
    configure_group(&mut group);

    for &n in &[12, 16, 20] {
        let circuit = circuits::hardware_efficient_ansatz(n, 3, SEED);

        group.bench_with_input(BenchmarkId::new("statevector", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Statevector, circ, 42).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("factored", n), &circuit, |b, circ| {
            b.iter(|| run_with(BackendKind::Factored, circ, 42).unwrap());
        });
    }

    group.finish();
}

fn clifford_t_circuit(n_qubits: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];

    // Clifford layer first
    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n_qubits - 1 {
        if rng.random_bool(0.5) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    // Insert T gates on random qubits
    for _ in 0..t_count {
        let q = rng.random_range(0..n_qubits);
        circuit.add_gate(Gate::T, &[q]);
    }

    // More Clifford layers
    for _ in 0..3 {
        for q in 0..n_qubits {
            let gate_idx = rng.random_range(0..cliffords.len());
            circuit.add_gate(cliffords[gate_idx].clone(), &[q]);
        }
        for q in 0..n_qubits - 1 {
            if rng.random_bool(0.5) {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    circuit
}

fn stabilizer_rank_terminal_shot_circuit(n_qubits: usize, t_count: usize, seed: u64) -> Circuit {
    let mut circuit = clifford_t_circuit(n_qubits, t_count, seed);
    let measured = n_qubits.min(4);
    circuit.num_classical_bits = measured;
    for q in 0..measured {
        circuit.add_measure(q, q);
    }
    circuit
}

fn stabilizer_rank_mid_circuit_shot_circuit(n_qubits: usize, t_count: usize, seed: u64) -> Circuit {
    let mut circuit = clifford_t_circuit(n_qubits, t_count, seed);
    circuit.num_classical_bits = 2;
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[1]),
    });
    circuit.add_reset(0);
    circuit.add_measure(1, 1);
    circuit
}

fn bench_stabilizer_rank_shot_case(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    n_qubits: usize,
    t_count: usize,
    shot_count: usize,
) {
    let branch_count = 1usize << t_count;
    let terminal = stabilizer_rank_terminal_shot_circuit(n_qubits, t_count, SEED);
    let id = format!("{n_qubits}q_chi{branch_count}_{shot_count}shots");
    group.bench_function(BenchmarkId::new("shots_terminal", &id), |b| {
        b.iter(|| {
            run_shots_with(
                BackendKind::StabilizerRank,
                black_box(&terminal),
                black_box(shot_count),
                SEED,
            )
            .unwrap();
        });
    });

    let mid = stabilizer_rank_mid_circuit_shot_circuit(n_qubits, t_count, SEED);
    group.bench_function(BenchmarkId::new("shots_mid_circuit", &id), |b| {
        b.iter(|| {
            run_shots_with(
                BackendKind::StabilizerRank,
                black_box(&mid),
                black_box(shot_count),
                SEED,
            )
            .unwrap();
        });
    });
}

fn bench_clifford_t(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford_t");
    configure_group(&mut group);

    let n = 10;
    for &t in &[1, 2, 4, 8, 12] {
        let circuit = clifford_t_circuit(n, t, SEED);

        group.bench_function(BenchmarkId::new("spd", format!("{n}q_{t}t")), |b| {
            b.iter(|| {
                prism_q::run_spd(&circuit, 0.0, 0).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("spp_10k", format!("{n}q_{t}t")), |b| {
            b.iter(|| {
                prism_q::run_spp(&circuit, 10_000, 42).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("statevector", format!("{n}q_{t}t")), |b| {
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    for &n in &[5, 15, 20] {
        let circuit = clifford_t_circuit(n, 4, SEED);

        group.bench_function(BenchmarkId::new("spd", format!("{n}q_4t")), |b| {
            b.iter(|| {
                prism_q::run_spd(&circuit, 0.0, 0).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("statevector", format!("{n}q_4t")), |b| {
            b.iter(|| {
                run_with(BackendKind::Statevector, &circuit, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_stabilizer_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("stabilizer_rank");
    configure_group(&mut group);

    // Compare stabilizer_rank exact vs SPD vs statevector
    let n = 10;
    for &t in &[2, 4, 8, 12] {
        let circuit = clifford_t_circuit(n, t, SEED);
        let id = format!("{n}q_{t}t");

        group.bench_function(BenchmarkId::new("stab_rank", &id), |b| {
            b.iter(|| {
                prism_q::run_stabilizer_rank(&circuit, 42).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("spd", &id), |b| {
            b.iter(|| {
                prism_q::run_spd(&circuit, 0.0, 0).unwrap();
            });
        });
    }

    // Approximate mode: higher T-counts with bounded terms
    for &t in &[16, 20] {
        let circuit = clifford_t_circuit(n, t, SEED);
        let id = format!("{n}q_{t}t");

        group.bench_function(BenchmarkId::new("approx_256", &id), |b| {
            b.iter(|| {
                prism_q::run_stabilizer_rank_approx(&circuit, 256, 42).unwrap();
            });
        });

        group.bench_function(BenchmarkId::new("approx_1024", &id), |b| {
            b.iter(|| {
                prism_q::run_stabilizer_rank_approx(&circuit, 1024, 42).unwrap();
            });
        });
    }

    let shot_count = 64;
    for &n in &[32, 64, 128] {
        for &t in &[1, 2, 3, 4] {
            bench_stabilizer_rank_shot_case(&mut group, n, t, shot_count);
        }
    }
    bench_stabilizer_rank_shot_case(&mut group, 1000, 1, shot_count);

    group.finish();
}

fn bench_compiled_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_sampler");
    configure_group(&mut group);

    for &n in &[100, 500, 1000] {
        let mut circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let id = format!("noiseless_{}q_10k", n);
        group.bench_function(BenchmarkId::new("noiseless", &id), |b| {
            b.iter(|| {
                prism_q::run_shots_compiled(&circuit, 10_000, SEED).unwrap();
            });
        });

        let id_lut = format!("lut_only_{}q_10k", n);
        group.bench_function(BenchmarkId::new("lut_only", &id_lut), |b| {
            let mut sampler = prism_q::compile_forward(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_bulk(10_000));
        });

        let id_packed = format!("packed_{}q_10k", n);
        group.bench_function(BenchmarkId::new("packed", &id_packed), |b| {
            let mut sampler = prism_q::compile_forward(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_bulk_packed(10_000));
        });

        let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let id_noisy = format!("noisy_{}q_10k", n);
        group.bench_function(BenchmarkId::new("noisy", &id_noisy), |b| {
            b.iter(|| {
                prism_q::run_shots_noisy(&circuit, &noise, 10_000, SEED).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_noisy_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("noisy_sampling");
    configure_group(&mut group);

    let clifford = with_terminal_measurements(circuits::clifford_heavy_circuit(100, 10, SEED));
    let clifford_noise = prism_q::NoiseModel::uniform_depolarizing(&clifford, 0.001);
    group.bench_function(
        BenchmarkId::new("compiled_pauli", "clifford_100q_10k"),
        |b| {
            b.iter(|| {
                run_shots_with_noise(BackendKind::Auto, &clifford, &clifford_noise, 10_000, SEED)
                    .unwrap();
            });
        },
    );

    let non_clifford = non_clifford_noise_circuit(12, 4);
    let non_clifford_noise = prism_q::NoiseModel::uniform_depolarizing(&non_clifford, 0.001);
    group.bench_function(
        BenchmarkId::new("trajectory_pauli", "non_clifford_12q_512"),
        |b| {
            b.iter(|| {
                run_shots_with_noise(
                    BackendKind::Auto,
                    &non_clifford,
                    &non_clifford_noise,
                    512,
                    SEED,
                )
                .unwrap();
            });
        },
    );

    // The correlated two-qubit branch, the only trajectory path that reads a
    // two-qubit reduced density matrix and applies the selected operator as a
    // `Fused2q`. One `2^n` reduction plus one `2^n` gate pass per event, plus a
    // boxed 4x4 per event, against the single gate pass a Pauli branch costs.
    // Same circuit shape and shot count as `trajectory_pauli`, so the pair is
    // comparable; the noise differs and nothing else does.
    let correlated_noise = prism_q::NoiseBuilder::new()
        .after_gates_joint(
            prism_q::GateFilter::all().named("cx"),
            correlated_zz_channel(0.001),
        )
        .build(&non_clifford)
        .unwrap();
    group.bench_function(
        BenchmarkId::new("trajectory_kraus_2q", "non_clifford_12q_512"),
        |b| {
            b.iter(|| {
                run_shots_with_noise(
                    BackendKind::Statevector,
                    &non_clifford,
                    &correlated_noise,
                    512,
                    SEED,
                )
                .unwrap();
            });
        },
    );

    group.finish();
}

fn bench_compiled_sampler_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_sampler_scale");
    configure_group(&mut group);

    let n = 100;
    let mut circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
    circuit.num_classical_bits = n;
    for i in 0..n {
        circuit.add_measure(i, i);
    }

    for &shots in &[100_000, 1_000_000, 10_000_000] {
        let label = format!("packed_100q_{shots}");
        group.bench_function(BenchmarkId::new("packed", &label), |b| {
            let mut sampler = prism_q::compile_forward(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_bulk_packed(shots));
        });
    }

    group.finish();
}

fn bench_compiled_sampler_filtered(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_sampler_filtered");
    configure_group(&mut group);

    for &n_pairs in &[50, 250, 500] {
        let circuit = compiled_filtered_bell_pairs_circuit(n_pairs);
        let n = circuit.num_qubits;

        let compile_id = format!("bell_pairs_{n}q");
        group.bench_function(BenchmarkId::new("compile", &compile_id), |b| {
            b.iter(|| prism_q::compile_measurements(&circuit, SEED).unwrap());
        });

        let packed_id = format!("bell_pairs_{n}q_10k");
        group.bench_function(BenchmarkId::new("packed", &packed_id), |b| {
            let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_bulk_packed(10_000));
        });

        let counts_id = format!("bell_pairs_{n}q_10k");
        group.bench_function(BenchmarkId::new("counts", &counts_id), |b| {
            let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
            b.iter(|| sampler.sample_counts(10_000));
        });
    }

    group.finish();
}

fn bench_spp(c: &mut Criterion) {
    let mut group = c.benchmark_group("spp");
    configure_group(&mut group);

    let num_samples = 10_000;

    for &t in &[2, 4, 8, 12] {
        let circuit = clifford_t_circuit(10, t, SEED);
        let id = format!("10q_{t}t");

        group.bench_function(BenchmarkId::new("spp_10k", &id), |b| {
            b.iter(|| {
                prism_q::sim::unified_pauli::run_spp(&circuit, num_samples, 42).unwrap();
            });
        });

        if t <= 12 {
            group.bench_function(BenchmarkId::new("spd", &id), |b| {
                b.iter(|| {
                    prism_q::run_spd(&circuit, 0.0, 0).unwrap();
                });
            });
        }
    }

    for &n in &[20, 50, 100] {
        let circuit = clifford_t_circuit(n, 8, SEED);
        let id = format!("{n}q_8t");

        group.bench_function(BenchmarkId::new("spp_10k", &id), |b| {
            b.iter(|| {
                prism_q::sim::unified_pauli::run_spp(&circuit, num_samples, 42).unwrap();
            });
        });
    }

    let circuit_100t = clifford_t_circuit(20, 100, SEED);
    group.bench_function(BenchmarkId::new("spp_10k", "20q_100t"), |b| {
        b.iter(|| {
            prism_q::sim::unified_pauli::run_spp(&circuit_100t, num_samples, 42).unwrap();
        });
    });

    let circuit_1000t = clifford_t_circuit(50, 1000, SEED);
    group.bench_function(BenchmarkId::new("spp_10k", "50q_1000t"), |b| {
        b.iter(|| {
            prism_q::sim::unified_pauli::run_spp(&circuit_1000t, num_samples, 42).unwrap();
        });
    });

    group.finish();
}

fn bench_coalesce_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalesce_baseline");
    configure_group(&mut group);

    let num_samples = 10_000;

    for &(n, depth, t_frac, label) in &[
        (10, 20, 0.05, "10q_d20_t5pct"),
        (10, 50, 0.05, "10q_d50_t5pct"),
        (20, 20, 0.05, "20q_d20_t5pct"),
        (50, 20, 0.05, "50q_d20_t5pct"),
        (100, 10, 0.05, "100q_d10_t5pct"),
    ] {
        let circuit = circuits::clifford_t_circuit(n, depth, t_frac, SEED);

        group.bench_function(BenchmarkId::new("spp_10k", label), |b| {
            b.iter(|| {
                prism_q::sim::unified_pauli::run_spp(&circuit, num_samples, 42).unwrap();
            });
        });
    }

    group.finish();
}

// ---- Density matrix (explicit backend, constructed directly) ----

fn run_dm_apply_only(circuit: &Circuit) {
    let mut backend = DensityMatrixBackend::new(42);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    black_box(backend.probabilities().unwrap());
}

fn bench_density_matrix_unitary_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_matrix/unitary_layers");
    configure_group(&mut group);

    for &n in &[4, 8, 10, 12] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                run_dm_apply_only(circ);
            });
        });
    }

    group.finish();
}

/// The same layers through the `Simulate` terminal, the only density-matrix
/// entry point that runs `fuse_circuit`.
///
/// `density_matrix/unitary_layers` calls `apply_instructions` directly and so
/// never reaches the fusion pipeline whatever `supports_fused_gates` returns.
/// Widths bracket the fusion constants: 8 is below `MIN_QUBITS_FOR_FUSION` and
/// fuses nothing, which makes it the in-group control; 10 admits one-qubit
/// fusion only; 12 adds the two-qubit and tiled passes.
fn bench_density_matrix_fused_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_matrix/fused_layers");
    configure_group(&mut group);

    for &n in &[8, 10, 12] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                black_box(
                    sim::simulate(circ)
                        .backend(BackendKind::DensityMatrix)
                        .seed(SEED)
                        .run()
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

/// The two noisy-sampling routes over the same circuit and noise model, swept
/// over width and shot count.
///
/// One exact density-matrix evolution answers any shot count, so the exact
/// route is near flat in shots and grows as `4^n`; the trajectory route
/// re-simulates per shot, so it is linear in shots and grows as `2^n`. Two
/// shot counts per width give both the intercept and the slope, which is what
/// a routing rule keyed on width alone cannot have.
fn bench_density_matrix_exact_vs_trajectory(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_matrix/exact_vs_trajectory");
    configure_group(&mut group);

    for &n in &[8usize, 10, 12] {
        let circuit = non_clifford_noise_circuit(n, 4);
        let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.001);
        for &shots in &[256usize, 4096] {
            for (label, kind) in [
                ("exact", BackendKind::DensityMatrix),
                ("trajectory", BackendKind::Statevector),
            ] {
                group.bench_function(BenchmarkId::new(label, format!("{n}q_{shots}")), |b| {
                    b.iter(|| {
                        run_shots_with_noise(kind.clone(), &circuit, &noise, shots, SEED).unwrap();
                    });
                });
            }
        }
    }

    group.finish();
}

/// Kraus set for amplitude damping at rate `gamma`.
fn amplitude_damping_kraus(gamma: f64) -> Vec<[[Complex64; 2]; 2]> {
    let c = |re: f64| Complex64::new(re, 0.0);
    let zero = c(0.0);
    vec![
        [[c(1.0), zero], [zero, c((1.0 - gamma).sqrt())]],
        [[zero, c(gamma.sqrt())], [zero, zero]],
    ]
}

/// Kraus set for symmetric one-qubit depolarizing at rate `p`.
fn depolarizing_kraus(p: f64) -> Vec<[[Complex64; 2]; 2]> {
    let c = |re: f64| Complex64::new(re, 0.0);
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

/// Kraus pair for correlated `ZZ` dephasing at rate `p`, in the
/// `2*bit(q0) + bit(q1)` packing `NoiseChannel::Kraus2q` takes.
fn correlated_zz_kraus(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let zero = Complex64::new(0.0, 0.0);
    let diag = |scale: f64, signs: [f64; 4]| {
        let mut m = [[zero; 4]; 4];
        for (t, sign) in signs.iter().enumerate() {
            m[t][t] = Complex64::new(scale * sign, 0.0);
        }
        m
    };
    vec![
        diag((1.0 - p).sqrt(), [1.0, 1.0, 1.0, 1.0]),
        diag(p.sqrt(), [1.0, -1.0, -1.0, 1.0]),
    ]
}

fn correlated_zz_channel(p: f64) -> prism_q::NoiseChannel {
    prism_q::NoiseChannel::Kraus2q {
        kraus: correlated_zz_kraus(p),
    }
}

/// The correlated `ZZ` pair conjugated by `H` on `q0`: `{sqrt(1-p) I, sqrt(p) X(x)Z}`.
/// Unitary conjugation of the whole set preserves CPTP, and the `X` factor fills the
/// compiled 16x16 superoperator, where the `ZZ` pair compiles to its diagonal.
fn h_conjugated_zz_kraus(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let zero = Complex64::new(0.0, 0.0);
    let mut k0 = [[zero; 4]; 4];
    let mut k1 = [[zero; 4]; 4];
    for t in 0..4 {
        k0[t][t] = Complex64::new((1.0 - p).sqrt(), 0.0);
        let sign = if t & 1 == 0 { 1.0 } else { -1.0 };
        k1[t][t ^ 2] = Complex64::new(p.sqrt() * sign, 0.0);
    }
    vec![k0, k1]
}

fn dm_backend(n: usize) -> DensityMatrixBackend {
    let circuit = circuits::random_circuit(n, 2, SEED);
    let mut backend = DensityMatrixBackend::new(42);
    backend.init(n, 1).unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    backend
}

/// Channel, measurement, and diagnostic sweeps over the `4^n` buffer.
///
/// The channel mix is amplitude damping and symmetric depolarizing through
/// `apply_1q_kraus`, plus symmetric two-qubit depolarizing. Measure and reset
/// run at qubit 0 and at the top qubit, the two ends of the `2^(qubit+n)` row
/// stride. Widths stop at 12: `4^14` is 4.3 GB per buffer, too large to iterate.
fn bench_density_matrix_noisy_channels(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_matrix/noisy_channels");
    configure_group(&mut group);

    for &n in &[10, 12] {
        let amp_damp = amplitude_damping_kraus(0.05);
        group.bench_with_input(BenchmarkId::new("kraus_amp_damp", n), &n, |b, &n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_1q_kraus(0, &amp_damp));
        });

        let depol = depolarizing_kraus(0.02);
        group.bench_with_input(BenchmarkId::new("kraus_depolarizing", n), &n, |b, &n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_1q_kraus(n - 1, &depol));
        });

        group.bench_with_input(BenchmarkId::new("depolarizing_2q", n), &n, |b, &n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_2q_depolarizing(0, n - 1, 0.02));
        });

        // The general two-qubit channel: every set compiles to one 16x16
        // superoperator, and at these widths the compile is under 0.01% of the
        // row, so each row reads its sweep and not the operator count. The zz
        // pair compiles to a diagonal superoperator and takes the contiguous
        // one-multiply pass; the three pairs pin its position independence.
        let zz = correlated_zz_kraus(0.02);
        group.bench_with_input(BenchmarkId::new("kraus_2q_q0_qtop", n), &n, |b, &n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_2q_kraus(0, n - 1, &zz));
        });
        group.bench_with_input(BenchmarkId::new("kraus_2q_adjacent", n), &n, |b, &_n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_2q_kraus(2, 3, &zz));
        });
        group.bench_with_input(BenchmarkId::new("kraus_2q_mid", n), &n, |b, &_n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_2q_kraus(1, 2, &zz));
        });

        // The same channel with off-diagonal structure, which fills the
        // superoperator and holds the dense sweep: `(2, 3)` reads its
        // four-block step and `(0, n-1)` the one-block fallback, where a
        // two-block step once measured +0.2% and +1.9% against one and was
        // dropped. The sweep's cost is data-independent, so these compare
        // directly against the zz rows at the same pairs.
        let hzz = h_conjugated_zz_kraus(0.02);
        group.bench_with_input(BenchmarkId::new("kraus_2q_dense", n), &n, |b, &_n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_2q_kraus(2, 3, &hzz));
        });
        group.bench_with_input(BenchmarkId::new("kraus_2q_dense_q0", n), &n, |b, &n| {
            let mut backend = dm_backend(n);
            b.iter(|| backend.apply_2q_kraus(0, n - 1, &hzz));
        });

        // A `Fused2q` that no `Multi2q` batch absorbs, which is what fusion
        // emits for an isolated two-qubit run. It takes the bra register
        // through `conjugate_gate`; without that arm it costs two extra
        // full-buffer conjugations, and no circuit row reaches it because the
        // brick layers in `random_circuit` batch every run.
        group.bench_with_input(BenchmarkId::new("fused_2q_gate", n), &n, |b, &_n| {
            let mut backend = dm_backend(n);
            let fused = Instruction::Gate {
                gate: Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())),
                targets: SmallVec::from_slice(&[0, 1]),
            };
            b.iter(|| backend.apply(&fused).unwrap());
        });

        for &(qubit, label) in &[(0usize, "measure_q0"), (n - 1, "measure_qtop")] {
            group.bench_with_input(BenchmarkId::new(label, n), &n, |b, &n| {
                let mut backend = dm_backend(n);
                let measure = Instruction::Measure {
                    qubit,
                    classical_bit: 0,
                };
                b.iter(|| backend.apply(&measure).unwrap());
            });
        }

        for &(qubit, label) in &[(0usize, "reset_q0"), (n - 1, "reset_qtop")] {
            group.bench_with_input(BenchmarkId::new(label, n), &n, |b, &n| {
                let mut backend = dm_backend(n);
                b.iter(|| backend.reset(qubit).unwrap());
            });
        }

        group.bench_with_input(BenchmarkId::new("purity", n), &n, |b, &n| {
            let backend = dm_backend(n);
            b.iter(|| black_box(backend.purity()));
        });
    }

    group.finish();
}

/// The exact noisy route, reachable through `Simulate` since the noise model
/// reached this backend. One mixed-state evolution then a draw per shot, so the
/// row tracks the `4^n` sweep count and is nearly flat in the shot count.
fn bench_density_matrix_noisy_shots(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_matrix/noisy_shots");
    configure_group(&mut group);

    for &n in &[8, 10] {
        let mut circuit = circuits::random_circuit(n, 4, SEED);
        circuit.num_classical_bits = n;
        for q in 0..n {
            circuit.add_measure(q, q);
        }
        let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                run_shots_with_noise(BackendKind::DensityMatrix, &circuit, &noise, 1024, SEED)
                    .unwrap();
            });
        });
    }

    group.finish();
}

/// Clifford layers over `n` qubits, emitted either as bare instructions or as
/// one guarded region per layer.
///
/// The guard holds for every layer, so both forms execute the same gates.
fn layered_clifford_body(circuit: &mut Circuit, n: usize, depth: usize, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];
    for layer in 0..depth {
        for q in 0..n {
            let pick = rng.random_range(0..cliffords.len());
            circuit.add_gate(cliffords[pick].clone(), &[q]);
        }
        for q in ((layer % 2)..n - 1).step_by(2) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
}

fn guarded_layers_circuit(n: usize, depth: usize, guard: Option<ClassicalCondition>) -> Circuit {
    let mut circuit = Circuit::new(n, 1);
    circuit.add_measure(0, 0);
    for layer in 0..depth {
        let mut body = Circuit::new(n, 1);
        layered_clifford_body(&mut body, n, 1, SEED ^ layer as u64);
        match &guard {
            Some(condition) => {
                if let Some(region) =
                    prism_q::circuit::guarded(condition.clone(), body.instructions)
                {
                    circuit.instructions.push(region);
                }
            }
            None => circuit.instructions.extend(body.instructions),
        }
    }
    circuit
}

/// What a guard costs: identical Clifford layers, once wrapped in a per-layer
/// guarded region that always holds and once emitted bare.
///
/// The statevector pair is dominated by fusion rather than by the branch, since
/// no pass fuses inside a region body or across its boundary. The `nofuse_`
/// pair on the stabilizer backend, which declines fused gates so the pass is
/// skipped on both sides, is the one that prices the branch alone.
fn bench_dynamic_guarded_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic/guarded_region");
    configure_group(&mut group);

    for &n in &[16usize, 20] {
        let taken = ClassicalCondition::BitIsZero(0);
        let guarded_circuit = guarded_layers_circuit(n, 20, Some(taken));
        let plain_circuit = guarded_layers_circuit(n, 20, None);

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| run_with(BackendKind::Statevector, &guarded_circuit, SEED).unwrap());
        });
        group.bench_function(format!("unguarded_{n}"), |b| {
            b.iter(|| run_with(BackendKind::Statevector, &plain_circuit, SEED).unwrap());
        });

        // The stabilizer backend declines fused gates, so the fusion pass is
        // skipped for both forms and the pair prices the branch alone.
        group.bench_function(format!("nofuse_{n}"), |b| {
            b.iter(|| run_with(BackendKind::Stabilizer, &guarded_circuit, SEED).unwrap());
        });
        group.bench_function(format!("nofuse_unguarded_{n}"), |b| {
            b.iter(|| run_with(BackendKind::Stabilizer, &plain_circuit, SEED).unwrap());
        });
    }

    group.finish();
}

/// A region guarded on a bit no preceding measurement writes, so it can never
/// be taken. Prices the coarse sampling predicate: the `absent` row is the same
/// circuit with the region deleted.
fn dead_region_circuit(n: usize, with_region: bool) -> Circuit {
    let mut circuit = Circuit::new(n, n);
    if with_region {
        let mut body = Circuit::new(n, n);
        body.add_gate(Gate::X, &[0]);
        body.add_gate(Gate::Cx, &[0, 1]);
        if let Some(region) =
            prism_q::circuit::guarded(ClassicalCondition::BitIsOne(n - 1), body.instructions)
        {
            circuit.instructions.push(region);
        }
    }
    layered_clifford_body(&mut circuit, n, 10, SEED);
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    circuit
}

fn bench_dynamic_dead_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic/dead_region");
    configure_group(&mut group);

    for &n in &[16usize, 20] {
        let with_region = dead_region_circuit(n, true);
        let without_region = dead_region_circuit(n, false);

        group.bench_function(format!("{n}"), |b| {
            b.iter(|| run_shots_with(BackendKind::Auto, &with_region, 1_000, SEED).unwrap());
        });
        group.bench_function(format!("absent_{n}"), |b| {
            b.iter(|| run_shots_with(BackendKind::Auto, &without_region, 1_000, SEED).unwrap());
        });
    }

    group.finish();
}

/// A region that genuinely branches on a measured bit, so the run replays once
/// per shot. The `terminal` row is the same circuit with the region deleted,
/// which samples one evolved distribution.
fn shot_cliff_circuit(n: usize, with_region: bool) -> Circuit {
    let mut circuit = Circuit::new(n, n);
    layered_clifford_body(&mut circuit, n, 6, SEED);
    circuit.add_measure(0, 0);
    if with_region {
        let mut body = Circuit::new(n, n);
        body.add_gate(Gate::X, &[1]);
        body.add_gate(Gate::Cx, &[1, 2]);
        if let Some(region) =
            prism_q::circuit::guarded(ClassicalCondition::BitIsOne(0), body.instructions)
        {
            circuit.instructions.push(region);
        }
    }
    for q in 1..n {
        circuit.add_measure(q, q);
    }
    circuit
}

fn bench_dynamic_shots(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic/shots");
    configure_group(&mut group);

    let n = 16;
    let with_region = shot_cliff_circuit(n, true);
    let without_region = shot_cliff_circuit(n, false);

    for &shots in &[1_000usize, 10_000] {
        group.bench_function(format!("{shots}"), |b| {
            b.iter(|| run_shots_with(BackendKind::Auto, &with_region, shots, SEED).unwrap());
        });
        group.bench_function(format!("terminal_{shots}"), |b| {
            b.iter(|| run_shots_with(BackendKind::Auto, &without_region, shots, SEED).unwrap());
        });
    }

    group.finish();
}

/// Neutrality row: an untouched statevector row re-run under a density-matrix
/// group name. The density-matrix backend shares no kernels with the
/// statevector path, so this must stay within the 5% regression gate.
fn bench_density_matrix_neutrality(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_matrix/neutrality");
    configure_group(&mut group);

    let circuit = circuits::qft_circuit(22);
    group.bench_with_input(BenchmarkId::from_parameter(22), &circuit, |b, circ| {
        b.iter(|| {
            run_with(BackendKind::Statevector, circ, 42).unwrap();
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets =
        // Statevector sweeps
    bench_statevector_random,
    bench_statevector_qft,
    bench_statevector_qft_textbook,
    bench_statevector_qpe,
    bench_statevector_hea,
    bench_statevector_qaoa,
    bench_statevector_trotter,
    bench_statevector_diag_mixed,
    bench_statevector_qv,
    bench_statevector_w_state,
    bench_statevector_clifford,
    bench_statevector_depth_sweep,
    bench_statevector_entanglement,
    bench_statevector_scalability,
    // Stabilizer
    bench_stabilizer_scaling,
    bench_stabilizer_measurement,
    // Factored stabilizer
    bench_factored_stabilizer_scaling,
    bench_factored_stabilizer_local,
    bench_factored_stabilizer_measurement,
    // Sparse
    bench_sparse_scaling,
    bench_sparse_low_entanglement,
    bench_sparse_sampling,
    // MPS
    bench_mps_scaling,
    bench_mps_linear_chain,
    bench_mps_hotspots,
    bench_mps_sampling,
    // Product state
    bench_product_scaling,
    bench_product_sampling,
    // Tensor network
    bench_tn_scaling,
    bench_tn_linear_chain,
    bench_tn_scalar_expectation,
    bench_tn_scalar_depth,
    bench_tn_scalar_wide_deep,
    bench_tn_scalar_tree_quality,
    bench_tn_rdm_chain,
    bench_tn_midmeasure_chain,
    bench_tn_noisy_chain,
    // Auto dispatch
    bench_auto_random,
    bench_auto_qft,
    bench_auto_qft_textbook,
    bench_auto_qpe,
    bench_auto_hea,
    bench_auto_clifford,
    bench_auto_scalability,
    // Cross-backend comparisons
    bench_compare_clifford,
    bench_compare_single_qubit,
    bench_compare_general,
    // Decomposition
    bench_decomposition,
    // Parameter sweep setup (rebuild vs rebind)
    bench_parameter_sweep,
    // Adjoint gradient (adjoint vs finite-difference per parameter)
    bench_gradient,
    bench_gradient_qaoa,
    bench_gradient_prefix,
    // Variational loop iteration (rebuild vs rebind under simulation cost)
    bench_vqe_loop,
    // Forward Pauli-sum expectation (parallel-sandwich neutrality)
    bench_expectation,
    bench_expectation_factored,
    bench_expectation_native,
    bench_expectation_density_matrix,
    bench_expectation_grouped,
    // Factored backend
    bench_factored_random,
    bench_factored_independent,
    bench_factored_sim_only,
    bench_factored_dynamic,
    bench_factored_partial_independence,
    bench_factored_dense,
    bench_factored_noise_kraus,
    // Clifford+T (SPD/SPP)
    bench_clifford_t,
    // Stabilizer rank
    bench_stabilizer_rank,
    // Compiled sampler (noiseless + noisy shot sampling)
    bench_compiled_sampler,
    // Noisy trajectory dispatch and compiled Pauli sampling
    bench_noisy_sampling,
    // Compiled sampler at scale (high shot counts)
    bench_compiled_sampler_scale,
    // Filtered compiled sampler (independent subsystem path)
    bench_compiled_sampler_filtered,
    // Stochastic Pauli Propagation (Clifford+T)
    bench_spp,
    // Coalescing baseline (interleaved Clifford+T)
    bench_coalesce_baseline,
    // Density matrix (explicit backend)
    bench_density_matrix_unitary_layers,
    bench_density_matrix_fused_layers,
    bench_density_matrix_exact_vs_trajectory,
    bench_density_matrix_noisy_channels,
    bench_density_matrix_noisy_shots,
    bench_density_matrix_neutrality,
    // Dynamic circuits (guard cost, dead-region predicate, per-shot cliff)
    bench_dynamic_guarded_region,
    bench_dynamic_dead_region,
    bench_dynamic_shots
}
criterion_main!(benches);
