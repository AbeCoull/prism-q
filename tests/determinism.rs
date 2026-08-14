//! Thread-count determinism contract from `docs/architecture/threading-simd.md`:
//! bitwise equality on the deterministic-partitioning paths, seeded sampling
//! identity, and the 1e-12 stability bound on parallel reductions.

#![cfg(feature = "parallel")]

mod common;

use common::SEED;
use num_complex::Complex64;
use prism_q::circuits::qft_circuit;
use prism_q::{
    BackendKind, Circuit, Gate, McuData, PauliTerm, StatevectorBackend, run_on, run_on_state,
    run_shots_compiled, simulate,
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

const REDUCTION_EPS: f64 = 1e-12;

fn in_pool<T: Send>(threads: usize, op: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
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
// batch tiers, plus Swap, Cu, and both Mcu shapes, which fuse into nothing and
// keep their own index-bijection kernels.
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
