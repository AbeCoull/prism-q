//! `Backend::reduced_density_matrix` and the `Simulate` terminal over it:
//! closed forms, agreement with the statevector over the shared small-circuit
//! corpus, the row-index convention, the trajectory routines at one and two
//! qubits, and subsystem validation. The cap is pinned in `export_cap_oversize.rs`.

use crate::common;

use common::circuits::{CircuitCase, exact_small_cases, product_separable_cases};
use common::{DM_EPS, FACTORED_EPS, MPS_EPS, PRODUCT_EPS, SEED, SPARSE_EPS, SV_EPS, TN_EPS};
use num_complex::Complex64;
use prism_q::PrismError;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits::{
    brickwork_circuit, cz_chain_circuit, ghz_circuit, hardware_efficient_ansatz, random_circuit,
};
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::{BackendKind, GateFilter, NoiseBuilder, NoiseChannel, simulate};

const EPS: f64 = 1e-12;

fn assert_density(rho: &[Complex64], k: usize, label: &str) {
    let dim = 1usize << k;
    assert_eq!(rho.len(), dim * dim, "{label}: side");
    let trace: Complex64 = (0..dim).map(|t| rho[t * dim + t]).sum();
    assert!(
        (trace.re - 1.0).abs() < EPS && trace.im.abs() < EPS,
        "{label}: trace {trace}"
    );
    for t in 0..dim {
        for tp in 0..dim {
            let (a, b) = (rho[t * dim + tp], rho[tp * dim + t].conj());
            assert!(
                (a - b).norm() < EPS,
                "{label}: entry ({t}, {tp}) {a} against {b} across the diagonal"
            );
        }
    }
}

fn assert_matrix_close(actual: &[Complex64], expected: &[Complex64], eps: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() < eps,
            "{label}: entry {i} reads {a} against {e}"
        );
    }
}

/// The partial trace summed directly over the amplitudes, scaled to trace
/// one: the reference every kernel is checked against.
fn naive_rdm(psi: &[Complex64], subsystem: &[usize]) -> Vec<Complex64> {
    let dim = 1usize << subsystem.len();
    let row = |idx: usize| {
        subsystem
            .iter()
            .enumerate()
            .fold(0, |t, (i, &q)| t | (((idx >> q) & 1) << i))
    };
    let named = subsystem.iter().fold(0usize, |m, &q| m | (1 << q));
    let mut rho = vec![Complex64::new(0.0, 0.0); dim * dim];
    for (i, a) in psi.iter().enumerate() {
        for (j, b) in psi.iter().enumerate() {
            if i & !named == j & !named {
                rho[row(i) * dim + row(j)] += a * b.conj();
            }
        }
    }
    let trace: f64 = (0..dim).map(|t| rho[t * dim + t].re).sum();
    for entry in &mut rho {
        *entry /= trace;
    }
    rho
}

fn rdm_of(backend: &mut dyn Backend, subsystem: &[usize]) -> Vec<Complex64> {
    let rho = backend.reduced_density_matrix(subsystem).unwrap();
    assert_density(
        &rho,
        subsystem.len(),
        &format!("{} on {subsystem:?}", backend.name()),
    );
    rho
}

fn rdm_on(backend: &mut dyn Backend, circuit: &Circuit, subsystem: &[usize]) -> Vec<Complex64> {
    sim::run_on(backend, circuit).unwrap();
    let rho = backend.reduced_density_matrix(subsystem).unwrap();
    assert_density(
        &rho,
        subsystem.len(),
        &format!("{} on {subsystem:?}", backend.name()),
    );
    rho
}

/// Every backend that answers a general (entangling) circuit.
fn entangling_backends() -> [Box<dyn Backend>; 5] {
    [
        Box::new(StatevectorBackend::new(SEED)),
        Box::new(SparseBackend::new(SEED)),
        Box::new(FactoredBackend::new(SEED)),
        Box::new(DensityMatrixBackend::new(SEED)),
        Box::new(TensorNetworkBackend::new(SEED)),
    ]
}

/// One-, two- and three-qubit subsystems of an `n`-qubit register, spread
/// across it and once more in reverse, so the row-index order is exercised.
fn subsystems(n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for k in 1..=3.min(n) {
        let step = if n >= 2 * k { 2 } else { 1 };
        let spread: Vec<usize> = (0..k).map(|i| i * step).collect();
        let reversed: Vec<usize> = spread.iter().rev().copied().collect();
        if reversed != spread {
            out.push(reversed);
        }
        out.push(spread);
    }
    out
}

/// Both the statevector kernel and the backend under test against the naive
/// sum over the exported amplitudes, so neither is its own oracle.
fn assert_matches_statevector(
    new_backend: &dyn Fn() -> Box<dyn Backend>,
    cases: &[CircuitCase],
    eps: f64,
) {
    for case in cases {
        let circuit = case.circuit();
        for subsystem in subsystems(circuit.num_qubits) {
            let mut sv = StatevectorBackend::new(SEED);
            let from_kernel = rdm_on(&mut sv, &circuit, &subsystem);
            let expected = naive_rdm(&sv.export_statevector().unwrap(), &subsystem);
            assert_matrix_close(
                &from_kernel,
                &expected,
                SV_EPS,
                &format!("statevector {} on {subsystem:?}", case.name),
            );
            let mut backend = new_backend();
            let actual = rdm_on(backend.as_mut(), &circuit, &subsystem);
            assert_matrix_close(
                &actual,
                &expected,
                eps,
                &format!("{} {} on {subsystem:?}", backend.name(), case.name),
            );
        }
    }
}

// Tracing either qubit of (|00> + |11>) / sqrt 2 leaves I / 2.
#[test]
fn bell_one_qubit_rdm_is_half_the_identity() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let half = Complex64::new(0.5, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    for qubit in 0..2 {
        for mut backend in entangling_backends() {
            let rho = rdm_on(backend.as_mut(), &circuit, &[qubit]);
            assert_matrix_close(&rho, &[half, zero, zero, half], EPS, backend.name());
        }
    }
}

// Any k qubits of (|00000> + |11111>) / sqrt 2 read
// (|0^k><0^k| + |1^k><1^k|) / 2: the two corners, nothing else.
#[test]
fn ghz_5_rdm_is_the_two_branch_mixture_at_k_1_2_3() {
    let circuit = ghz_circuit(5);
    for subsystem in [vec![2usize], vec![4, 1], vec![0, 3, 2]] {
        let dim = 1usize << subsystem.len();
        let mut expected = vec![Complex64::new(0.0, 0.0); dim * dim];
        expected[0] = Complex64::new(0.5, 0.0);
        expected[dim * dim - 1] = Complex64::new(0.5, 0.0);
        for mut backend in entangling_backends() {
            let rho = rdm_on(backend.as_mut(), &circuit, &subsystem);
            assert_matrix_close(
                &rho,
                &expected,
                EPS,
                &format!("{} on {subsystem:?}", backend.name()),
            );
        }
    }
}

/// The `n`-qubit W state by a cascade of controlled rotations: at step `i`
/// the excitation sits on `q[i]` with weight `(n - i) / n`, a controlled
/// `Ry` on `q[i + 1]` leaves `1 / n` of it there, and a `CX` back moves the
/// rest along. The controlled `Ry(theta)` is `Ry(theta / 2)`, `CX`,
/// `Ry(-theta / 2)`, `CX` on the target.
fn w_state(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    circuit.add_gate(Gate::X, &[0]);
    for i in 0..n - 1 {
        let theta = 2.0 * (1.0 / (n - i) as f64).sqrt().acos();
        circuit.add_gate(Gate::Ry(theta / 2.0), &[i + 1]);
        circuit.add_gate(Gate::Cx, &[i, i + 1]);
        circuit.add_gate(Gate::Ry(-theta / 2.0), &[i + 1]);
        circuit.add_gate(Gate::Cx, &[i, i + 1]);
        circuit.add_gate(Gate::Cx, &[i + 1, i]);
    }
    circuit
}

// W on four qubits, (|0001> + |0010> + |0100> + |1000>) / 2. One qubit is
// diag(3/4, 1/4). Two qubits: the two branches with the excitation outside
// the pair give |00><00| / 2, the two with it inside give
// (|01> + |10>)(<01| + <10|) / 4, in the pair's own bit order.
#[test]
fn w_4_rdm_on_one_and_two_qubits() {
    let circuit = w_state(4);
    let c = |re: f64| Complex64::new(re, 0.0);
    let one_qubit = [c(0.75), c(0.0), c(0.0), c(0.25)];
    let mut two_qubits = vec![c(0.0); 16];
    two_qubits[0] = c(0.5);
    for (t, tp) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
        two_qubits[t * 4 + tp] = c(0.25);
    }
    for mut backend in entangling_backends() {
        let rho = rdm_on(backend.as_mut(), &circuit, &[3]);
        assert_matrix_close(&rho, &one_qubit, EPS, backend.name());
        let rho = rdm_on(backend.as_mut(), &circuit, &[2, 0]);
        assert_matrix_close(&rho, &two_qubits, EPS, backend.name());
    }
}

// S H |0> = (|0> + i|1>) / sqrt 2 reads [[1/2, -i/2], [i/2, 1/2]]: the
// imaginary entry sits at <1|rho|0>, and a swapped ket and bra conjugates it.
#[test]
fn s_h_reads_an_imaginary_off_diagonal() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::S, &[1]);
    let expected = [
        Complex64::new(0.5, 0.0),
        Complex64::new(0.0, -0.5),
        Complex64::new(0.0, 0.5),
        Complex64::new(0.5, 0.0),
    ];
    let backends = entangling_backends().into_iter().chain([
        Box::new(MpsBackend::new(SEED, 4)) as Box<dyn Backend>,
        Box::new(ProductStateBackend::new(SEED)),
        Box::new(StabilizerBackend::new(SEED)),
        Box::new(FactoredStabilizerBackend::new(SEED)),
    ]);
    for mut backend in backends {
        let rho = rdm_on(backend.as_mut(), &circuit, &[1]);
        assert_matrix_close(&rho, &expected, EPS, backend.name());
    }
}

#[test]
fn sparse_matches_the_statevector_on_the_small_corpus() {
    assert_matches_statevector(
        &|| Box::new(SparseBackend::new(SEED)),
        &exact_small_cases(),
        SPARSE_EPS,
    );
}

#[test]
fn factored_matches_the_statevector_on_the_small_corpus() {
    assert_matches_statevector(
        &|| Box::new(FactoredBackend::new(SEED)),
        &exact_small_cases(),
        FACTORED_EPS,
    );
}

// The chain sweeps its own environment, so the cost follows the span the named
// qubits occupy and the reversed subsystems pin the permutation back into the
// requested order.
#[test]
fn mps_matches_the_statevector_on_the_small_corpus() {
    assert_matches_statevector(
        &|| Box::new(MpsBackend::new(SEED, 64)),
        &exact_small_cases(),
        MPS_EPS,
    );
}

// The eight-qubit cases put the mixture past the parallel threshold, so both
// partial-trace paths of the density matrix are covered.
#[test]
fn density_matrix_matches_the_statevector_on_the_small_corpus() {
    assert_matches_statevector(
        &|| Box::new(DensityMatrixBackend::new(SEED)),
        &exact_small_cases(),
        DM_EPS,
    );
}

#[test]
fn tensor_network_matches_the_statevector_on_the_small_corpus() {
    assert_matches_statevector(
        &|| Box::new(TensorNetworkBackend::new(SEED)),
        &exact_small_cases(),
        TN_EPS,
    );
}

#[test]
fn product_matches_the_statevector_on_the_separable_corpus() {
    assert_matches_statevector(
        &|| Box::new(ProductStateBackend::new(SEED)),
        &product_separable_cases(),
        PRODUCT_EPS,
    );
}

// The general kernel against the tuned one- and two-qubit trajectory
// routines, whose two-qubit packing puts `q0` in the high bit: below the
// parallel threshold at eight qubits, where the sums run in one order and
// agree to rounding, and above it at fifteen. Unit-norm states only, since
// the trajectory routines report the stored norm where the general kernel
// scales to trace one.
#[test]
fn one_and_two_qubit_rows_agree_with_the_trajectory_routines() {
    for (n, eps) in [(8usize, 1e-15), (15, 1e-13)] {
        let circuit = random_circuit(n, 4, SEED);
        let mut backend = StatevectorBackend::new(SEED);
        sim::run_on(&mut backend, &circuit).unwrap();
        for q in 0..n {
            let one = backend.reduced_density_matrix_1q(q).unwrap();
            let flat: Vec<Complex64> = one.iter().flatten().copied().collect();
            let rho = backend.reduced_density_matrix(&[q]).unwrap();
            assert_matrix_close(&rho, &flat, eps, &format!("{n} qubits, qubit {q}"));
        }
        for q0 in 0..n {
            for q1 in (0..n).filter(|&q1| q1 != q0) {
                let two = backend.reduced_density_matrix_2q(q0, q1).unwrap();
                let flat: Vec<Complex64> = two.iter().flatten().copied().collect();
                let rho = backend.reduced_density_matrix(&[q1, q0]).unwrap();
                assert_matrix_close(&rho, &flat, eps, &format!("{n} qubits, pair {q0} {q1}"));
            }
        }
    }
}

// |+> on q0 and |1> on q1. With `subsystem = [0, 1]` the row index is
// `2 * q1 + q0`, so the weight sits on rows 2 and 3; with `[1, 0]` it is
// `2 * q0 + q1`, rows 1 and 3. Every answering backend pins the same order.
#[test]
fn the_row_index_takes_subsystem_0_as_its_lowest_bit() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::X, &[1]);
    let half = Complex64::new(0.5, 0.0);
    let expect = |rows: [usize; 2]| {
        let mut rho = vec![Complex64::new(0.0, 0.0); 16];
        for t in rows {
            for tp in rows {
                rho[t * 4 + tp] = half;
            }
        }
        rho
    };
    let backends: [Box<dyn Backend>; 6] = [
        Box::new(StatevectorBackend::new(SEED)),
        Box::new(SparseBackend::new(SEED)),
        Box::new(FactoredBackend::new(SEED)),
        Box::new(DensityMatrixBackend::new(SEED)),
        Box::new(ProductStateBackend::new(SEED)),
        Box::new(TensorNetworkBackend::new(SEED)),
    ];
    for mut backend in backends {
        let rho = rdm_on(backend.as_mut(), &circuit, &[0, 1]);
        assert_matrix_close(&rho, &expect([2, 3]), EPS, backend.name());
        let rho = rdm_on(backend.as_mut(), &circuit, &[1, 0]);
        assert_matrix_close(&rho, &expect([1, 3]), EPS, backend.name());
    }
}

// Tracing nothing out returns the buffer itself to rounding of the trace
// scale, or its rows and columns permuted when the subsystem is not in
// register order; the pure case is the outer product of the exported
// amplitudes.
#[test]
fn the_whole_register_returns_the_state_itself() {
    let circuit = random_circuit(4, 3, SEED);
    let mut dm = DensityMatrixBackend::new(SEED);
    sim::run_on(&mut dm, &circuit).unwrap();
    let buffer = dm.density_matrix().unwrap();
    let whole = dm.reduced_density_matrix(&[0, 1, 2, 3]).unwrap();
    assert_matrix_close(&whole, &buffer, DM_EPS, "density matrix at k = n");

    let order = [3usize, 1, 0, 2];
    let permuted = dm.reduced_density_matrix(&order).unwrap();
    let index = |t: usize| {
        order
            .iter()
            .enumerate()
            .fold(0, |i, (bit, &q)| i | (((t >> bit) & 1) << q))
    };
    for t in 0..16 {
        for tp in 0..16 {
            let (a, e) = (permuted[t * 16 + tp], buffer[index(t) * 16 + index(tp)]);
            assert!(
                (a - e).norm() < DM_EPS,
                "permuted entry ({t}, {tp}) {a} against {e}"
            );
        }
    }

    let mut sv = StatevectorBackend::new(SEED);
    let rho = rdm_on(&mut sv, &circuit, &[0, 1, 2, 3]);
    let psi = sv.export_statevector().unwrap();
    let outer: Vec<Complex64> = (0..16)
        .flat_map(|t| (0..16).map(move |tp| (t, tp)))
        .map(|(t, tp)| psi[t] * psi[tp].conj())
        .collect();
    assert_matrix_close(&rho, &outer, SV_EPS, "statevector at k = n");
}

// The naive sum against the threaded partial trace at fifteen qubits of a
// brick-wall state, whose off-diagonals are complex, with the subsystem
// spread across the register: three qubits take the fold over the traced
// index, twelve the stripes of result rows.
#[test]
fn the_parallel_partial_trace_matches_a_naive_sum() {
    let n = 15;
    let circuit = brickwork_circuit(n, 6, SEED);
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();
    let psi = sv.export_statevector().unwrap();
    for subsystem in [vec![11usize, 2, 7], (0..12).map(|i| (7 * i) % 15).collect()] {
        let rho = rdm_of(&mut sv, &subsystem);
        let expected = naive_rdm(&psi, &subsystem);
        assert_matrix_close(
            &rho,
            &expected,
            SV_EPS,
            &format!("statevector at 15 qubits on {subsystem:?}"),
        );
    }
}

/// Two brick-wall blocks on `q[0..4]` and `q[4..8]` that never meet, with
/// `Rz` phases so the off-diagonals are complex.
fn two_block_brickwork() -> Circuit {
    let mut circuit = Circuit::new(8, 0);
    for layer in 0..4 {
        for q in 0..8 {
            circuit.add_gate(Gate::Ry(0.3 + 0.41 * q as f64 + 0.17 * layer as f64), &[q]);
            circuit.add_gate(Gate::Rz(0.9 * q as f64 - 0.23 * layer as f64), &[q]);
        }
        let pairs: &[[usize; 2]] = if layer % 2 == 0 {
            &[[0, 1], [2, 3], [4, 5], [6, 7]]
        } else {
            &[[1, 2], [5, 6]]
        };
        for pair in pairs {
            circuit.add_gate(Gate::Cz, pair);
        }
    }
    circuit
}

// A subsystem taking two qubits from each block, interleaved, so the block
// factors are multiplied in through non-adjacent bits; the reference is the
// naive sum over the statevector's amplitudes.
#[test]
fn factored_blocks_compose_against_a_naive_sum() {
    let circuit = two_block_brickwork();
    let subsystem = [5usize, 1, 6, 2];
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();
    let expected = naive_rdm(&sv.export_statevector().unwrap(), &subsystem);
    let rho = rdm_on(&mut FactoredBackend::new(SEED), &circuit, &subsystem);
    assert_matrix_close(&rho, &expected, FACTORED_EPS, "factored on two blocks");
}

// A state whose norm is not one: a brick-wall state shrunk by diag(1, 1/2)
// on one qubit, the same on a separable state so the product backend takes
// part, and a statevector after two measurements, which sets its pending
// norm. Every answer is trace one and agrees with the naive sum over the
// exported amplitudes, which carries the same norm.
#[test]
fn the_trace_is_one_whatever_norm_the_state_carries() {
    let shrink = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
    ];
    let subsystem = [4usize, 2, 0];
    let mut separable = Circuit::new(6, 0);
    for q in 0..6 {
        separable.add_gate(Gate::Ry(0.4 * q as f64 + 0.3), &[q]);
        separable.add_gate(Gate::Rz(0.7 * q as f64 - 0.2), &[q]);
    }
    let fixtures: [(Circuit, Vec<Box<dyn Backend>>); 2] = [
        (
            brickwork_circuit(6, 4, 21),
            entangling_backends().into_iter().collect(),
        ),
        (
            separable,
            entangling_backends()
                .into_iter()
                .chain([Box::new(ProductStateBackend::new(SEED)) as Box<dyn Backend>])
                .collect(),
        ),
    ];
    for (circuit, backends) in fixtures {
        let mut sv = StatevectorBackend::new(SEED);
        sim::run_on(&mut sv, &circuit).unwrap();
        sv.apply_1q_matrix(2, &shrink).unwrap();
        let expected = naive_rdm(&sv.export_statevector().unwrap(), &subsystem);
        for mut backend in backends {
            sim::run_on(backend.as_mut(), &circuit).unwrap();
            backend.apply_1q_matrix(2, &shrink).unwrap();
            let rho = rdm_of(backend.as_mut(), &subsystem);
            assert_matrix_close(&rho, &expected, 1e-14, backend.name());
        }
    }

    let mut measured = brickwork_circuit(6, 4, 21);
    measured.num_classical_bits = 2;
    measured.add_measure(1, 0);
    measured.add_measure(3, 1);
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &measured).unwrap();
    let rho = rdm_of(&mut sv, &subsystem);
    let expected = naive_rdm(&sv.export_statevector().unwrap(), &subsystem);
    assert_matrix_close(&rho, &expected, 1e-14, "statevector after two measurements");
}

// Both tableau backends read the marginal off the projector onto the
// generators supported inside the subsystem, never off a dense export.
#[test]
fn the_tableau_backends_answer_the_reduced_density_matrix() {
    let circuit = ghz_circuit(4);
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, &circuit).unwrap();
    let expected = naive_rdm(&sv.export_statevector().unwrap(), &[0, 2]);
    let backends: [Box<dyn Backend>; 2] = [
        Box::new(StabilizerBackend::new(SEED)),
        Box::new(FactoredStabilizerBackend::new(SEED)),
    ];
    for mut backend in backends {
        sim::run_on(backend.as_mut(), &circuit).unwrap();
        let name = backend.name();
        let rho = rdm_of(backend.as_mut(), &[0, 2]);
        assert_matrix_close(&rho, &expected, 1e-14, name);
    }
}

// The same rejections on every backend that answers, and unlike a cut, the
// whole register is a valid subsystem.
#[test]
fn a_subsystem_is_a_non_empty_set_of_distinct_qubits_up_to_the_whole_register() {
    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::Ry(0.5 * q as f64 + 0.2), &[q]);
    }
    let backends: [Box<dyn Backend>; 6] = [
        Box::new(StatevectorBackend::new(SEED)),
        Box::new(SparseBackend::new(SEED)),
        Box::new(FactoredBackend::new(SEED)),
        Box::new(DensityMatrixBackend::new(SEED)),
        Box::new(ProductStateBackend::new(SEED)),
        Box::new(TensorNetworkBackend::new(SEED)),
    ];
    for mut backend in backends {
        sim::run_on(backend.as_mut(), &circuit).unwrap();
        let name = backend.name();
        assert!(
            matches!(
                backend.reduced_density_matrix(&[]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took an empty subsystem"
        );
        assert!(
            matches!(
                backend.reduced_density_matrix(&[1, 1]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took a repeated qubit"
        );
        assert_eq!(
            backend.reduced_density_matrix(&[0, 4]).unwrap_err(),
            PrismError::InvalidQubit {
                index: 4,
                register_size: 4
            },
            "{name} took an out-of-range qubit"
        );
        let rho = backend.reduced_density_matrix(&[0, 1, 2, 3]).unwrap();
        assert_density(&rho, 4, name);
    }
}

/// The terminal's matrix for one explicit backend kind, with the subsystem it
/// echoes back checked on the way through.
fn terminal_rdm(kind: BackendKind, circuit: &Circuit, qubits: &[usize]) -> Vec<Complex64> {
    let result = simulate(circuit)
        .backend(kind)
        .seed(SEED)
        .reduced_density_matrix(qubits)
        .unwrap();
    assert_eq!(
        result.qubits, qubits,
        "the terminal echoed a different subsystem"
    );
    result.data
}

/// The terminal on each backend it can route to, against the statevector
/// kernel, which the naive sums above pin in turn.
fn assert_terminal_matches_statevector(rows: &[(BackendKind, f64)], cases: &[CircuitCase]) {
    for case in cases {
        let circuit = case.circuit();
        let mut sv = StatevectorBackend::new(SEED);
        sim::run_on(&mut sv, &circuit).unwrap();
        for subsystem in subsystems(circuit.num_qubits) {
            let expected = sv.reduced_density_matrix(&subsystem).unwrap();
            for (kind, eps) in rows {
                let label = format!("{kind:?} {} on {subsystem:?}", case.name);
                let rho = terminal_rdm(kind.clone(), &circuit, &subsystem);
                assert_density(&rho, subsystem.len(), &label);
                assert_matrix_close(&rho, &expected, *eps, &label);
            }
        }
    }
}

#[test]
fn the_terminal_matches_the_statevector_on_the_small_corpus() {
    assert_terminal_matches_statevector(
        &[
            (BackendKind::Statevector, SV_EPS),
            (BackendKind::Sparse, SPARSE_EPS),
            (BackendKind::Factored, FACTORED_EPS),
            (BackendKind::Mps { max_bond_dim: 64 }, MPS_EPS),
            (BackendKind::DensityMatrix, DM_EPS),
            (BackendKind::TensorNetwork, TN_EPS),
        ],
        &exact_small_cases(),
    );
}

#[test]
fn the_terminal_matches_the_statevector_on_the_separable_corpus() {
    assert_terminal_matches_statevector(
        &[(BackendKind::ProductState, PRODUCT_EPS)],
        &product_separable_cases(),
    );
}

// Past the naive sum's reach, so the reference is the statevector kernel the
// naive sums above pin. Reversed pairs and spread triples exercise the
// permutation out of the contraction's leg order.
#[test]
fn tensor_network_matches_the_statevector_at_20_qubits() {
    let subsystems: [&[usize]; 6] = [
        &[10],
        &[3, 16],
        &[16, 3],
        &[9, 10],
        &[0, 9, 19],
        &[12, 4, 8],
    ];
    for circuit in [
        cz_chain_circuit(20, 6, SEED),
        hardware_efficient_ansatz(20, 2, SEED),
    ] {
        let mut sv = StatevectorBackend::new(SEED);
        sim::run_on(&mut sv, &circuit).unwrap();
        let mut tn = TensorNetworkBackend::new(SEED);
        sim::run_on(&mut tn, &circuit).unwrap();
        for subsystem in subsystems {
            let expected = sv.reduced_density_matrix(subsystem).unwrap();
            let rho = rdm_of(&mut tn, subsystem);
            assert_matrix_close(&rho, &expected, TN_EPS, &format!("{subsystem:?}"));
        }
    }
}

/// Apply `circuit` without a run, which would read probabilities past their cap.
fn load_wide(backend: &mut dyn Backend, circuit: &Circuit) {
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
}

// Any k qubits of the 40-qubit GHZ state read the two corners at 1/2.
#[test]
fn ghz_40_pair_and_triple_are_the_two_branch_mixture() {
    let mut tn = TensorNetworkBackend::new(SEED);
    load_wide(&mut tn, &ghz_circuit(40));
    for subsystem in [vec![0usize, 39], vec![25, 7], vec![3, 20, 38]] {
        let dim = 1usize << subsystem.len();
        let mut expected = vec![Complex64::new(0.0, 0.0); dim * dim];
        expected[0] = Complex64::new(0.5, 0.0);
        expected[dim * dim - 1] = Complex64::new(0.5, 0.0);
        let rho = rdm_of(&mut tn, &subsystem);
        assert_matrix_close(&rho, &expected, EPS, &format!("{subsystem:?}"));
    }
}

// H and CX on each rung (2j, 2j + 1) of 40 qubits: a rung reads
// |Phi+><Phi+|, four entries of 1/2 at rows 0 and 3, and a pair taken from two
// rungs reads I / 4.
#[test]
fn bell_ladder_40_reads_a_pure_pair_or_the_identity() {
    let mut circuit = Circuit::new(40, 0);
    for j in 0..20 {
        circuit.add_gate(Gate::H, &[2 * j]);
        circuit.add_gate(Gate::Cx, &[2 * j, 2 * j + 1]);
    }
    let mut tn = TensorNetworkBackend::new(SEED);
    load_wide(&mut tn, &circuit);
    let c = |re: f64| Complex64::new(re, 0.0);
    let mut rung = vec![c(0.0); 16];
    for (t, tp) in [(0, 0), (0, 3), (3, 0), (3, 3)] {
        rung[t * 4 + tp] = c(0.5);
    }
    let mut straddle = vec![c(0.0); 16];
    for t in 0..4 {
        straddle[t * 4 + t] = c(0.25);
    }
    for (subsystem, expected) in [
        ([12usize, 13], &rung),
        ([39, 38], &rung),
        ([13, 14], &straddle),
        ([0, 39], &straddle),
    ] {
        let rho = rdm_of(&mut tn, &subsystem);
        assert_matrix_close(&rho, expected, EPS, &format!("{subsystem:?}"));
    }
}

// Depth 4 holds the chain's bond at 4, so a 64 bond MPS is exact and stands in
// for the statevector at 40 qubits. Once through the backend and once through
// the terminal, which runs the fused gate tensors.
#[test]
fn tensor_network_pair_matches_an_untruncated_chain_at_40_qubits() {
    let circuit = cz_chain_circuit(40, 4, SEED);
    let mut mps = MpsBackend::new(SEED, 64);
    load_wide(&mut mps, &circuit);
    let mut tn = TensorNetworkBackend::new(SEED);
    load_wide(&mut tn, &circuit);
    for subsystem in [[19usize, 20], [20, 19], [5, 30]] {
        let expected = mps.reduced_density_matrix(&subsystem).unwrap();
        let label = format!("{subsystem:?}");
        assert_matrix_close(&rdm_of(&mut tn, &subsystem), &expected, TN_EPS, &label);
        let from_terminal = terminal_rdm(BackendKind::TensorNetwork, &circuit, &subsystem);
        assert_matrix_close(&from_terminal, &expected, TN_EPS, &label);
    }
}

// Depolarizing noise on a Bell pair leaves the pair mixed, so the two-qubit
// marginal of the exact mixture has purity below the pure state's 1 while
// still reading trace one.
#[test]
fn the_terminal_reads_the_exact_mixture_under_a_noise_model() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = NoiseBuilder::new()
        .after_gates(GateFilter::all(), NoiseChannel::Depolarizing { p: 0.05 })
        .build(&circuit)
        .unwrap();
    let result = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .reduced_density_matrix(&[0, 1])
        .unwrap();
    assert_density(&result.data, 2, "density matrix under noise");
    let purity = result.purity();
    assert!(
        (0.5..1.0).contains(&purity),
        "purity {purity} of a lightly depolarized Bell pair"
    );
}
