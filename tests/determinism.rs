//! Thread-count determinism contract from `docs/architecture/threading-simd.md`:
//! bitwise equality on the deterministic-partitioning paths, seeded sampling
//! identity, and the 1e-12 stability bound on parallel reductions.

#![cfg(feature = "parallel")]

mod common;

use common::{SEED, count_gates, mix_seed};
use num_complex::Complex64;
use prism_q::circuit::SmallVec;
use prism_q::circuits::qft_circuit;
use prism_q::{
    BackendKind, Circuit, ClassicalCondition, Gate, Instruction, McuData, NoiseModel, Parameters,
    PauliObservable, PauliTerm, PreparedCircuit, ResolvedBackend, StatevectorBackend, ThreadPool,
    run_on, run_on_state, run_shots_compiled, simulate,
};

#[cfg(not(miri))]
const THREADS_HI: usize = 4;
#[cfg(miri)]
const THREADS_HI: usize = 2;

// Miri sizes sit above the reduced parallel and fusion floors (8 under miri,
// see backend/mod.rs), so the same fused parallel kernels run on a state the
// interpreter can execute in minutes rather than hours.
#[cfg(not(miri))]
const DENSE_SIZES: &[usize] = &[10, 16, 18];
#[cfg(miri)]
const DENSE_SIZES: &[usize] = &[10];

#[cfg(not(miri))]
const QFT_SIZES: &[usize] = &[16, 20];
#[cfg(miri)]
const QFT_SIZES: &[usize] = &[10];

#[cfg(not(miri))]
const SAMPLING_QUBITS: usize = 16;
#[cfg(miri)]
const SAMPLING_QUBITS: usize = 10;

#[cfg(not(miri))]
const SAMPLING_SHOTS: usize = 4096;
#[cfg(miri)]
const SAMPLING_SHOTS: usize = 256;

// The size at which the dense circuit reaches every fused family named in
// `the_dense_circuit_still_reaches_the_fused_kernel_families`. Native needs 18
// for the diagonal batch floor; the miri floors all drop to 8, so 10 does it
// there.
#[cfg(not(miri))]
const FUSED_COVERAGE_QUBITS: usize = 18;
#[cfg(miri)]
const FUSED_COVERAGE_QUBITS: usize = 10;

const REDUCTION_EPS: f64 = 1e-12;

fn in_pool<T: Send>(threads: usize, op: impl FnOnce() -> T + Send) -> T {
    ThreadPool::with_threads(threads)
        .expect("scoped Rayon pool")
        .install(op)
}

fn rx_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    let (s, c) = (theta / 2.0).sin_cos();
    [
        [Complex64::new(c, 0.0), Complex64::new(0.0, -s)],
        [Complex64::new(0.0, -s), Complex64::new(c, 0.0)],
    ]
}

fn phase_matrix(theta: f64) -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta)],
    ]
}

// Non-Clifford circuit whose fused form walks the parallel kernel families:
// 1q runs, CX with absorbable neighbors, Rzz and phase runs for the diagonal
// batch tiers, the three Pauli-rotation branches (pair mix at low and high
// pivot, Z-parity diagonal), plus Swap, Cu, and both Mcu shapes, which fuse
// into nothing and keep their own index-bijection kernels.
fn representative_dense_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::Ry(0.31 + 0.07 * q as f64), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Rz(0.11 * (q + 1) as f64), &[q]);
        c.add_gate(Gate::Cx, &[q, q + 1]);
        c.add_gate(Gate::Rx(0.23 + 0.05 * q as f64), &[q + 1]);
    }
    for q in (0..n - 1).step_by(2) {
        c.add_gate(Gate::Rzz(0.17 + 0.03 * q as f64), &[q, q + 1]);
    }
    for q in 0..n {
        c.add_gate(Gate::P(0.05 + 0.02 * q as f64), &[q]);
    }
    c.add_pauli_rotation(0.19, &[PauliTerm::x(0), PauliTerm::y(2), PauliTerm::z(4)]);
    c.add_pauli_rotation(0.27, &[PauliTerm::z(1), PauliTerm::z(3), PauliTerm::z(5)]);
    c.add_pauli_rotation(0.33, &[PauliTerm::z(0), PauliTerm::x(n - 1)]);
    c.add_pauli_rotation(0.21, &[PauliTerm::x(n - 3), PauliTerm::x(n - 1)]);
    c.add_gate(Gate::Swap, &[0, n - 1]);
    c.add_gate(Gate::Cu(Box::new(rx_matrix(0.4))), &[1, n - 2]);
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: rx_matrix(0.7),
            num_controls: 2,
        })),
        &[0, 2, n - 3],
    );
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: phase_matrix(0.9),
            num_controls: 2,
        })),
        &[1, 3, n - 4],
    );
    c
}

fn amplitudes_with_threads(circuit: &Circuit, threads: usize) -> Vec<Complex64> {
    in_pool(threads, || {
        let mut backend = StatevectorBackend::new(SEED);
        run_on(&mut backend, circuit).expect("statevector run");
        backend.state_vector().to_vec()
    })
}

fn assert_bitwise_equal(base: &[Complex64], other: &[Complex64], label: &str) {
    assert_eq!(base.len(), other.len(), "{label}: length mismatch");
    for (idx, (a, b)) in base.iter().zip(other).enumerate() {
        assert!(
            a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits(),
            "{label}: amplitude {idx} differs bitwise: {a} vs {b}"
        );
    }
}

// Everything below asserts bitwise agreement, which unfused kernels satisfy
// just as well as fused ones, so a fusion floor moving out of reach would drain
// this file's coverage without turning a single test red. Pin the forms the
// circuit is built to produce instead. The three named here are the ones that
// survive both configurations: at the miri sizes the two-qubit runs are all
// swallowed by Multi2q, and a bare Fused2q appears only at the native width.
#[test]
fn the_dense_circuit_still_reaches_the_fused_kernel_families() {
    let circuit = representative_dense_circuit(FUSED_COVERAGE_QUBITS);
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    for (label, want) in [
        (
            "Fused",
            &(|g: &Gate| matches!(g, Gate::Fused(_))) as &dyn Fn(&Gate) -> bool,
        ),
        ("Multi2q", &|g: &Gate| matches!(g, Gate::Multi2q(_))),
        ("BatchRzz", &|g: &Gate| matches!(g, Gate::BatchRzz(_))),
    ] {
        assert!(
            count_gates(&fused, want) > 0,
            "the {FUSED_COVERAGE_QUBITS}q fused stream carries no {label}, so the              parallel kernel it stands for is no longer covered here"
        );
    }
}

#[test]
fn unitary_amplitudes_bitwise_equal_across_thread_counts() {
    for &n in DENSE_SIZES {
        let circuit = representative_dense_circuit(n);
        let base = amplitudes_with_threads(&circuit, 1);
        let wide = amplitudes_with_threads(&circuit, THREADS_HI);
        assert_bitwise_equal(&base, &wide, &format!("dense {n}q"));
    }
}

#[test]
fn qft_amplitudes_bitwise_equal_across_thread_counts() {
    for &n in QFT_SIZES {
        let circuit = qft_circuit(n);
        let dim = 1usize << n;
        let mut state: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64 * 0.7).sin() + 0.2, (i as f64 * 0.3).cos()))
            .collect();
        let norm = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for amp in &mut state {
            *amp /= norm;
        }

        let run = |threads: usize| {
            in_pool(threads, || {
                let mut backend = StatevectorBackend::new(SEED);
                run_on_state(&mut backend, &circuit, &state).expect("qft run");
                backend.state_vector().to_vec()
            })
        };
        assert_bitwise_equal(&run(1), &run(THREADS_HI), &format!("qft {n}q"));
    }
}

// Ignored under miri: four full circuit executions for kernels the other
// tests already interpret; the sampling walk itself is safe sequential code.
#[test]
#[cfg_attr(miri, ignore)]
fn terminal_shots_and_counts_identical_across_thread_counts() {
    let mut circuit = representative_dense_circuit(SAMPLING_QUBITS);
    circuit.measure_all();

    let shots = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .backend(BackendKind::Statevector)
                .seed(SEED)
                .shots(SAMPLING_SHOTS)
                .expect("shots")
                .shots
        })
    };
    assert_eq!(shots(1), shots(THREADS_HI), "terminal shots differ");

    let counts = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .backend(BackendKind::Statevector)
                .seed(SEED)
                .sample_counts(SAMPLING_SHOTS)
                .expect("counts")
                .counts
        })
    };
    assert_eq!(counts(1), counts(THREADS_HI), "terminal counts differ");
}

// Collapse probabilities come from a Rayon reduction whose combine order moves
// with the pool, so the post-measurement scale is ulp-stable, not bitwise.
// The seeded outcomes still agree unless a draw lands inside that ulp gap.
#[test]
fn midcircuit_collapse_ulp_stable_across_thread_counts() {
    let n = SAMPLING_QUBITS;
    let mut circuit = Circuit::new(n, 2);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::Ry(0.29 + 0.05 * q as f64), &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    circuit.add_measure(3, 0);
    circuit.add_reset(2);
    for q in 0..n {
        circuit.add_gate(Gate::Rx(0.13 + 0.04 * q as f64), &[q]);
    }
    circuit.add_measure(n - 1, 1);

    let run = |threads: usize| {
        in_pool(threads, || {
            let mut backend = StatevectorBackend::new(SEED);
            let outcome = run_on(&mut backend, &circuit).expect("collapse run");
            (
                outcome.classical_bits,
                outcome.probabilities.expect("dense probabilities").to_vec(),
            )
        })
    };
    let (bits_base, probs_base) = run(1);
    let (bits_wide, probs_wide) = run(THREADS_HI);

    assert_eq!(bits_base, bits_wide, "seeded collapse outcomes differ");
    let max_diff = probs_base
        .iter()
        .zip(&probs_wide)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_diff <= REDUCTION_EPS,
        "post-collapse probabilities differ by {max_diff:e}"
    );
}

// Ignored under miri: the reduction it bounds is safe code, and the collapse
// test interprets the same reduce shape at a fraction of the cost.
#[test]
#[cfg_attr(miri, ignore)]
fn expectation_values_ulp_stable_across_thread_counts() {
    let circuit = representative_dense_circuit(SAMPLING_QUBITS);
    let observables = vec![
        vec![PauliTerm::z(0), PauliTerm::z(5)],
        vec![PauliTerm::x(2)],
        vec![PauliTerm::y(1), PauliTerm::z(7), PauliTerm::x(9)],
    ];

    let run = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .backend(BackendKind::Statevector)
                .seed(SEED)
                .expectation_values(&observables)
                .expect("expectation values")
        })
    };
    let base = run(1);
    let wide = run(THREADS_HI);
    for (idx, (a, b)) in base.iter().zip(&wide).enumerate() {
        assert!(
            (a - b).abs() <= REDUCTION_EPS,
            "observable {idx} differs by {:e}",
            (a - b).abs()
        );
    }
}

// The grouped route reads a large group's mean and variance from a parallel
// moments reduction, on the state as run for a Z-only group and on a rotated
// copy otherwise.
#[test]
#[cfg_attr(miri, ignore)]
fn observable_expectation_ulp_stable_across_thread_counts() {
    let n = SAMPLING_QUBITS;
    let circuit = representative_dense_circuit(n);
    let chain = (0..n - 1).map(|q| (1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]));
    let field = (0..n).map(|q| (0.5, vec![PauliTerm::x(q)]));
    let pair = [(0.25, vec![PauliTerm::y(0), PauliTerm::y(3)])];
    let observable =
        PauliObservable::from_terms(chain.chain(field).chain(pair).collect::<Vec<_>>()).unwrap();

    let run = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .backend(BackendKind::Statevector)
                .seed(SEED)
                .observable_expectation(&observable)
                .expect("observable expectation")
        })
    };
    let base = run(1);
    let wide = run(THREADS_HI);
    assert!(
        (base.mean - wide.mean).abs() <= REDUCTION_EPS,
        "mean differs"
    );
    let (base_groups, wide_groups) = (base.group_variances.unwrap(), wide.group_variances.unwrap());
    for (idx, (a, b)) in base_groups.iter().zip(&wide_groups).enumerate() {
        assert!(
            (a - b).abs() <= REDUCTION_EPS,
            "group {idx} variance differs by {:e}",
            (a - b).abs()
        );
    }
}

// Below the kernels' parallel floor a prepared sweep splits bindings across
// workers, each on its own copy, so which worker takes a binding must not
// reach the result.
#[test]
#[cfg_attr(miri, ignore)]
fn prepared_sweeps_bitwise_equal_across_thread_counts() {
    let template = prism_q::circuits::hardware_efficient_ansatz(10, 2, SEED);
    let params = Parameters::all_rotations(&template);
    let points: Vec<Vec<f64>> = (0..16)
        .map(|k| {
            (0..params.num_slots())
                .map(|s| 0.37 * (k * params.num_slots() + s) as f64)
                .collect()
        })
        .collect();
    let observables: Vec<Vec<PauliTerm>> = (0..9)
        .map(|q| vec![PauliTerm::z(q), PauliTerm::x(q + 1)])
        .collect();

    let sweep = |threads: usize| {
        in_pool(threads, || {
            let mut prepared = PreparedCircuit::new(template.clone(), params.clone()).unwrap();
            let probabilities: Vec<Vec<f64>> = prepared
                .run_many(&points, SEED)
                .expect("run_many")
                .into_iter()
                .map(|outcome| outcome.probabilities.expect("probabilities").to_vec())
                .collect();
            let values = prepared
                .expectation_values_many(&points, &observables, SEED)
                .expect("expectation_values_many");
            (probabilities, values)
        })
    };
    let (base_probs, base_values) = sweep(1);
    let (wide_probs, wide_values) = sweep(THREADS_HI);
    assert_eq!(base_probs, wide_probs, "prepared run_many differs");
    assert_eq!(
        base_values, wide_values,
        "prepared expectation_values_many differs"
    );
}

// A parameter-shift gradient below the parallel floor splits its links into one
// chunk per worker, so the chunk boundaries move with the thread count.
#[test]
#[cfg_attr(miri, ignore)]
fn shift_gradient_bitwise_equal_across_thread_counts() {
    let circuit = prism_q::circuits::hardware_efficient_ansatz(10, 2, SEED);
    let params = Parameters::all_rotations(&circuit);
    let hamiltonian: Vec<(f64, Vec<PauliTerm>)> = (0..9)
        .map(|q| {
            (
                0.5 + 0.1 * q as f64,
                vec![PauliTerm::z(q), PauliTerm::x(q + 1)],
            )
        })
        .collect();
    let gradient = |threads: usize| {
        in_pool(threads, || {
            prism_q::run_expectation_gradient_shift(&circuit, &hamiltonian, &params, SEED)
                .expect("shift gradient")
        })
    };
    let base = gradient(1);
    for threads in [3, THREADS_HI] {
        assert_eq!(base, gradient(threads), "{threads} threads");
    }
}

// The batched compiled sampler derives one RNG stream per worker, so its shot
// set is a function of the pool width: reproducible at a fixed thread count,
// documented as thread-count-dependent in threading-simd.md.
#[test]
fn compiled_sampler_reproducible_at_fixed_thread_count() {
    let mut circuit = prism_q::circuits::ghz_circuit(12);
    circuit.measure_all();

    let run = || {
        in_pool(THREADS_HI, || {
            run_shots_compiled(&circuit, 256, SEED)
                .expect("compiled sampling")
                .shots
        })
    };
    assert_eq!(
        run(),
        run(),
        "same seed and pool width must reproduce shots"
    );
}

// The sparse sampler uses the MPS substream family (streams 2 and up) at
// shot-block grain; same pins and miri ignore as the MPS case below.
#[test]
#[cfg_attr(miri, ignore)]
fn sparse_terminal_shots_identical_across_thread_counts() {
    let mut circuit = prism_q::circuits::sparse_walk_circuit(20, 12, 2, SEED);
    circuit.measure_all();
    let kind = BackendKind::Sparse;

    let shots = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .backend(kind.clone())
                .seed(SEED)
                .shots(SAMPLING_SHOTS)
                .expect("shots")
                .shots
        })
    };
    let single = shots(1);
    assert_eq!(single.len(), SAMPLING_SHOTS);
    assert_eq!(single, shots(THREADS_HI), "sparse terminal shots differ");
    assert_eq!(single, shots(1), "sparse terminal shots not seed stable");

    // A shot count off the block grain leaves a partial trailing block.
    let odd = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .backend(kind.clone())
                .seed(SEED)
                .shots(SAMPLING_SHOTS - 37)
                .expect("shots")
                .shots
        })
    };
    assert_eq!(
        odd(1),
        odd(THREADS_HI),
        "sparse shots differ at a partial block"
    );

    let counts = in_pool(THREADS_HI, || {
        simulate(&circuit)
            .backend(kind.clone())
            .seed(SEED)
            .sample_counts(SAMPLING_SHOTS)
            .expect("counts")
            .counts
    });
    assert_eq!(
        counts.values().sum::<u64>(),
        SAMPLING_SHOTS as u64,
        "sparse counts do not sum to the shot count"
    );
}

// The MPS sampler draws each shot from a substream keyed on (seed, shot), so
// the words are identical at any thread count; the pins are same-seed
// stability and counts summing to shots, never a fixed bitstring. Ignored
// under miri like the dense sampling case above.
#[test]
#[cfg_attr(miri, ignore)]
fn mps_terminal_shots_identical_across_thread_counts() {
    let mut circuit = prism_q::circuits::brickwork_circuit(12, 8, SEED);
    circuit.measure_all();
    let kind = BackendKind::Mps { max_bond_dim: 64 };

    let shots = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .backend(kind.clone())
                .seed(SEED)
                .shots(SAMPLING_SHOTS)
                .expect("shots")
                .shots
        })
    };
    let single = shots(1);
    assert_eq!(single.len(), SAMPLING_SHOTS);
    assert_eq!(single, shots(THREADS_HI), "mps terminal shots differ");
    assert_eq!(single, shots(1), "mps terminal shots not seed stable");

    let counts = in_pool(THREADS_HI, || {
        simulate(&circuit)
            .backend(kind.clone())
            .seed(SEED)
            .sample_counts(SAMPLING_SHOTS)
            .expect("counts")
            .counts
    });
    assert_eq!(
        counts.values().sum::<u64>(),
        SAMPLING_SHOTS as u64,
        "mps counts do not sum to the shot count"
    );
}

const PER_SHOT_SHOTS: usize = 512;

// Rotations and CX on `qubits`, a measurement of the first into `bit`, then
// more rotations, so every shot replays the circuit.
fn add_mid_circuit_block(c: &mut Circuit, qubits: &[usize], bit: usize) {
    for (i, &q) in qubits.iter().enumerate() {
        c.add_gate(Gate::Ry(0.37 + 0.11 * i as f64), &[q]);
        c.add_gate(Gate::Rz(0.53 + 0.07 * i as f64), &[q]);
    }
    for pair in qubits.windows(2) {
        c.add_gate(Gate::Cx, pair);
    }
    c.add_measure(qubits[0], bit);
    for (i, &q) in qubits.iter().enumerate() {
        c.add_gate(Gate::Rx(0.29 + 0.13 * i as f64), &[q]);
    }
    for pair in qubits.windows(2).rev() {
        c.add_gate(Gate::Cx, pair);
    }
}

fn measure_every_qubit(c: &mut Circuit, first_bit: usize) {
    for q in 0..c.num_qubits {
        c.add_measure(q, first_bit + q);
    }
}

// Each shot runs on `mix_seed(SEED, i)` whether the loop splits or not, so the
// shots match a pool of one, a wider pool, and separate runs on those seeds.
fn assert_per_shot_matches_serial(circuit: &Circuit, route: ResolvedBackend) {
    let shots = |threads: usize| {
        in_pool(threads, || {
            simulate(circuit)
                .seed(SEED)
                .shots(PER_SHOT_SHOTS)
                .expect("shots")
        })
    };
    let single = shots(1);
    let wide = shots(THREADS_HI);
    assert_eq!(single.metadata.backend, route, "unexpected shot route");
    assert_eq!(single.shots, wide.shots, "per-shot bits differ");
    assert_eq!(
        format!("{:?}", single.metadata),
        format!("{:?}", wide.metadata),
        "per-shot metadata differs"
    );

    let separate: Vec<Vec<bool>> = (0..PER_SHOT_SHOTS)
        .map(|i| {
            simulate(circuit)
                .seed(mix_seed(SEED, i))
                .run()
                .expect("run")
                .classical_bits
        })
        .collect();
    assert_eq!(single.shots, separate, "shots differ from seeded runs");
}

#[test]
#[cfg_attr(miri, ignore)]
fn mid_circuit_shots_identical_across_thread_counts() {
    let n = 8;
    let mut circuit = Circuit::new(n, n + 1);
    add_mid_circuit_block(&mut circuit, &(0..n).collect::<Vec<_>>(), 0);
    measure_every_qubit(&mut circuit, 1);
    assert_per_shot_matches_serial(&circuit, ResolvedBackend::Statevector);
}

#[test]
#[cfg_attr(miri, ignore)]
fn decomposed_mid_circuit_shots_identical_across_thread_counts() {
    let n = 8;
    let mut circuit = Circuit::new(n, n + 2);
    add_mid_circuit_block(&mut circuit, &[0, 1, 2, 3], 0);
    add_mid_circuit_block(&mut circuit, &[4, 5, 6, 7], 1);
    measure_every_qubit(&mut circuit, 2);
    assert_per_shot_matches_serial(&circuit, ResolvedBackend::Decomposed);
}

// H and S layers over brick CX, then a measurement of qubit 0 that conditions
// an X on qubit 1 and a reset of qubit 0, so the circuit stays off the compiled
// sampler and every shot replays on a tableau.
fn dynamic_clifford_circuit(n: usize) -> Circuit {
    let layer = |c: &mut Circuit, depth: usize| {
        for q in 0..n {
            let gate = if (q + depth).is_multiple_of(3) {
                Gate::S
            } else {
                Gate::H
            };
            c.add_gate(gate, &[q]);
        }
        for q in ((depth % 2)..n - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    };
    let mut circuit = Circuit::new(n, n);
    for depth in 0..3 {
        layer(&mut circuit, depth);
    }
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[1]),
    });
    circuit.add_reset(0);
    for depth in 3..6 {
        layer(&mut circuit, depth);
    }
    measure_every_qubit(&mut circuit, 0);
    circuit
}

#[test]
#[cfg_attr(miri, ignore)]
fn stabilizer_mid_circuit_shots_identical_across_thread_counts() {
    let circuit = dynamic_clifford_circuit(20);
    assert_per_shot_matches_serial(&circuit, ResolvedBackend::Stabilizer);
}

#[test]
#[cfg_attr(miri, ignore)]
fn product_state_mid_circuit_shots_identical_across_thread_counts() {
    let n = 14;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(Gate::Ry(0.31 + 0.07 * q as f64), &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_measure(q, q);
        circuit.instructions.push(Instruction::Conditional {
            condition: ClassicalCondition::BitIsOne(q),
            gate: Gate::H,
            targets: SmallVec::from_slice(&[q + 1]),
        });
    }
    circuit.add_measure(n - 1, n - 1);
    assert_per_shot_matches_serial(&circuit, ResolvedBackend::ProductState);
}

#[test]
#[cfg_attr(miri, ignore)]
fn noisy_stabilizer_trajectories_identical_across_thread_counts() {
    let circuit = dynamic_clifford_circuit(20);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let shots = |threads: usize| {
        in_pool(threads, || {
            simulate(&circuit)
                .noise(&noise)
                .seed(SEED)
                .shots(PER_SHOT_SHOTS)
                .expect("noisy shots")
        })
    };
    let single = shots(1);
    let wide = shots(THREADS_HI);
    assert_eq!(
        single.metadata.backend,
        ResolvedBackend::Stabilizer,
        "unexpected trajectory route"
    );
    assert_eq!(single.shots, wide.shots, "trajectory bits differ");
    assert_eq!(
        format!("{:?}", single.metadata),
        format!("{:?}", wide.metadata),
        "trajectory metadata differs"
    );

    // Three shots stay on the serial loop, which must draw the same shot seeds.
    let serial = simulate(&circuit)
        .noise(&noise)
        .seed(SEED)
        .shots(3)
        .expect("serial noisy shots");
    assert_eq!(
        serial.shots,
        single.shots[..3],
        "serial trajectories differ from split ones"
    );
}
