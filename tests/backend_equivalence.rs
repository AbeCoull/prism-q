//! Cross-backend equivalence on hand-written circuits: gate kernels, the
//! parallel crossover at 16q and 20q, `export_statevector`, and subsystem
//! decomposition against a monolithic run. A comparison catches a divergence
//! but cannot say which side is wrong, so nothing here is an authority: the
//! closed-form values live in `tests/golden_small_circuits.rs`, and
//! `tests/conformance_matrix.rs` runs the same comparison over a generated
//! corpus with an explicit blame rule.

mod common;

use common::{SEED, TN_EPS, assert_probs_close, run_fused_probs};
use num_complex::Complex64;
use prism_q::Instruction;
use prism_q::PauliTerm;
use prism_q::backend::Backend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::Circuit;
use prism_q::gates::{Gate, McuData};
use prism_q::sim;

/// Tighter than `common::SV_EPS` and `common::MPS_EPS`, which are the bounds the
/// shared corpus needs at its largest sizes. The fixtures here are hand-written
/// and small, so they hold the exact-arithmetic bound.
const EPS: f64 = 1e-12;
const MPS_EPS: f64 = 1e-10;

/// Statevector switches to Rayon kernels at 14 qubits, so the 16q and 20q rows
/// compare a parallel run against a sequential backend. Reductions pair their
/// partial sums differently there, which costs a few ulp.
const PAR_EPS: f64 = 1e-10;

fn run_with(
    kind: prism_q::BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> prism_q::Result<prism_q::RunOutcome> {
    sim::simulate(circuit).backend(kind).seed(seed).run()
}

fn run_and_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut StatevectorBackend::new(SEED), circuit)
}

fn run_and_state(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.state_vector().to_vec()
}

fn run_stabilizer_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut StabilizerBackend::new(SEED), circuit)
}

fn run_sparse_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut SparseBackend::new(SEED), circuit)
}

fn run_mps_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut MpsBackend::new(SEED, 64), circuit)
}

fn run_tn_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut TensorNetworkBackend::new(SEED), circuit)
}

fn run_product_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut ProductStateBackend::new(SEED), circuit)
}

fn assert_probs(actual: &[f64], expected: &[f64]) {
    assert_probs_close(actual, expected, EPS, "statevector");
}

fn assert_probs_mps(actual: &[f64], expected: &[f64]) {
    assert_probs_close(actual, expected, MPS_EPS, "mps");
}

fn assert_probs_tn(actual: &[f64], expected: &[f64]) {
    assert_probs_close(actual, expected, TN_EPS, "tn");
}

fn assert_probs_par(actual: &[f64], expected: &[f64], label: &str) {
    assert_probs_close(actual, expected, PAR_EPS, label);
}

fn sx_sxdg_test_circuit() -> Circuit {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::SX, &[0]);
    c.add_gate(Gate::SXdg, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::SX, &[2]);
    c
}

fn p_theta_test_circuit() -> Circuit {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::P(std::f64::consts::FRAC_PI_4), &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::P(1.23), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::P(std::f64::consts::FRAC_PI_3), &[2]);
    c
}

fn cu_test_circuit() -> Circuit {
    let h_mat = Gate::H.matrix_2x2();
    let rz_mat = Gate::Rz(0.7).matrix_2x2();
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cu(Box::new(rz_mat)), &[1, 2]);
    c
}

#[test]
fn stabilizer_bell_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
}

#[test]
fn stabilizer_ghz4_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
}

#[test]
fn stabilizer_all_cliffords_match_statevector() {
    let gates = [
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
    ];
    for gate in &gates {
        let mut c = Circuit::new(1, 0);
        c.add_gate(gate.clone(), &[0]);
        assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
    }
}

#[test]
fn stabilizer_swap_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
}

#[test]
fn stabilizer_cz_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
}

#[test]
fn stabilizer_complex_clifford_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[2, 3]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Sdg, &[1]);
    assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
}

#[test]
fn sparse_bell_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn sparse_ghz4_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn sparse_all_single_gates_match_statevector() {
    let gates = [
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::Rx(1.234),
        Gate::Ry(2.345),
        Gate::Rz(3.456),
    ];
    for gate in &gates {
        let mut c = Circuit::new(1, 0);
        c.add_gate(gate.clone(), &[0]);
        assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
    }
}

#[test]
fn sparse_swap_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn sparse_cz_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn sparse_complex_circuit_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.5), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(1.0), &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[2, 3]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Rz(0.7), &[0]);
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_bell_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_ghz4_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_all_single_gates_match_statevector() {
    let gates = [
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::Rx(1.234),
        Gate::Ry(2.345),
        Gate::Rz(3.456),
    ];
    for gate in &gates {
        let mut c = Circuit::new(1, 0);
        c.add_gate(gate.clone(), &[0]);
        assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
    }
}

#[test]
fn mps_swap_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_cz_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_complex_circuit_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.5), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(1.0), &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[2, 3]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Rz(0.7), &[0]);
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_non_adjacent_cx_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 3]);
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_measurement_matches_statevector() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);

    let mut sv_backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv_backend, &c).unwrap();
    let sv_bits = sv_backend.classical_results().to_vec();

    let mut mps_backend = MpsBackend::new(SEED, 64);
    sim::run_on(&mut mps_backend, &c).unwrap();
    let mps_bits = mps_backend.classical_results().to_vec();

    assert_eq!(sv_bits, mps_bits);
}

#[test]
fn mcu_sparse_toffoli_matches_statevector() {
    let x_mat = Gate::X.matrix_2x2();
    let toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(toffoli, &[0, 1, 2]);
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn mcu_sparse_ccz_matches_statevector() {
    let z_mat = Gate::Z.matrix_2x2();
    let ccz = Gate::Mcu(Box::new(McuData {
        mat: z_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(ccz, &[0, 1, 3]);
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

// ---- CPhase cross-backend tests ----

#[test]
fn cphase_mps_matches_statevector() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_3), &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::cphase(0.7), &[1, 2]);
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn cphase_sparse_matches_statevector() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_3), &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::cphase(0.7), &[1, 2]);
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_matches_statevector_mcu() {
    let x_mat = Gate::X.matrix_2x2();
    let mcu = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(mcu, &[0, 1, 3]); // non-adjacent target
    assert_probs(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn product_h_all_qubits() {
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::H, &[q]);
    }
    assert_probs(&run_product_probs(&c), &run_and_probs(&c));
}

#[test]
fn product_rotation_circuit() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::Rx(0.7), &[0]);
    c.add_gate(Gate::Ry(1.3), &[1]);
    c.add_gate(Gate::Rz(2.1), &[2]);
    c.add_gate(Gate::Rx(0.4), &[3]);
    assert_probs(&run_product_probs(&c), &run_and_probs(&c));
}

#[test]
fn product_mixed_single_gates() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(Gate::Y, &[3]);
    c.add_gate(Gate::Rz(0.5), &[0]);
    assert_probs(&run_product_probs(&c), &run_and_probs(&c));
}

#[test]
fn tn_bell_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
}

#[test]
fn tn_ghz4_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
}

#[test]
fn tn_all_single_gates_match_statevector() {
    let gates = [
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::Rx(1.234),
        Gate::Ry(2.345),
        Gate::Rz(3.456),
    ];
    for gate in &gates {
        let mut c = Circuit::new(1, 0);
        c.add_gate(gate.clone(), &[0]);
        assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
    }
}

#[test]
fn tn_complex_circuit_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.5), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(1.0), &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[2, 3]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Rz(0.7), &[0]);
    assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
}

fn assert_rdm_matches_statevector(circuit: &Circuit) {
    let mut tn = TensorNetworkBackend::new(SEED);
    sim::run_on(&mut tn, circuit).unwrap();
    let mut sv = StatevectorBackend::new(SEED);
    sim::run_on(&mut sv, circuit).unwrap();

    for q in 0..circuit.num_qubits {
        let actual = tn.reduced_density_matrix_1q(q).unwrap();
        let expected = sv.reduced_density_matrix_1q(q).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (actual[row][col] - expected[row][col]).norm() < TN_EPS,
                    "tn rdm q{q}[{row}][{col}]: {} vs {}",
                    actual[row][col],
                    expected[row][col]
                );
            }
        }
    }
}

#[test]
fn tn_reduced_density_matrix_matches_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Ry(0.6), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Rx(1.1), &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::S, &[0]);
    assert_rdm_matches_statevector(&c);
}

// An untouched qubit leaves its ket and bra tensors sharing no leg, so the 2x2
// comes out of join_disjoint as an outer product rather than a contraction.
#[test]
fn tn_reduced_density_matrix_with_idle_qubit_matches_statevector() {
    let mut c = Circuit::new(5, 0);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Rz(0.9), &[2]);
    c.add_gate(Gate::Rx(0.4), &[4]);
    assert_rdm_matches_statevector(&c);
}

// The ancilla is prepared in |1> before it is measured, so both backends take
// the same branch and the tensor network answers from the dense tensor
// replace_with_statevector leaves behind.
#[test]
fn tn_reduced_density_matrix_after_measurement_matches_statevector() {
    let mut c = Circuit::new(3, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::X, &[2]);
    c.add_measure(2, 0);
    c.add_gate(Gate::Ry(0.8), &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    assert_rdm_matches_statevector(&c);
}

fn assert_pauli_expectations_match_statevector(circuit: &Circuit, observables: &[Vec<PauliTerm>]) {
    let mut tn = TensorNetworkBackend::new(SEED);
    sim::run_on(&mut tn, circuit).unwrap();

    let actual = tn.pauli_expectations(observables).unwrap();
    let expected = sim::run_expectation_values(circuit, observables, SEED).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - b).abs() < TN_EPS,
            "observable {i}: tn={a}, statevector={b}"
        );
    }
}

#[test]
fn tn_pauli_expectations_match_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Ry(0.6), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Rx(1.1), &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::S, &[0]);

    let observables = vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(2)],
        vec![PauliTerm::y(3)],
        vec![PauliTerm::z(0), PauliTerm::z(1)],
        vec![PauliTerm::x(1), PauliTerm::y(2)],
        vec![
            PauliTerm::y(0),
            PauliTerm::x(1),
            PauliTerm::z(2),
            PauliTerm::y(3),
        ],
        vec![],
    ];
    assert_pauli_expectations_match_statevector(&c, &observables);
}

// A weight-k observable appends k tensors and closes the other n-k legs
// directly, so an idle qubit exercises the elided-identity path.
#[test]
fn tn_pauli_expectations_with_idle_qubit_match_statevector() {
    let mut c = Circuit::new(5, 0);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Rz(0.9), &[2]);
    c.add_gate(Gate::Rx(0.4), &[4]);

    let observables = vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::y(4)],
        vec![PauliTerm::z(1), PauliTerm::z(2)],
        vec![PauliTerm::x(1), PauliTerm::x(2), PauliTerm::y(4)],
    ];
    assert_pauli_expectations_match_statevector(&c, &observables);
}

// The builder terminal rejected this backend before it answered observables
// natively, so the route matters as much as the trait method.
#[test]
fn tn_expectation_values_route_matches_auto() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Ry(0.6), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Rx(1.1), &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);

    let observables = vec![
        vec![PauliTerm::z(0), PauliTerm::z(1)],
        vec![PauliTerm::x(1), PauliTerm::y(2)],
        vec![PauliTerm::y(3)],
    ];

    let actual = sim::simulate(&c)
        .backend(prism_q::BackendKind::TensorNetwork)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();
    let expected = sim::run_expectation_values(&c, &observables, SEED).unwrap();
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - b).abs() < TN_EPS, "observable {i}: tn={a}, auto={b}");
    }
}

#[test]
fn tn_pauli_expectations_reject_duplicate_and_out_of_range_factors() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    let mut tn = TensorNetworkBackend::new(SEED);
    sim::run_on(&mut tn, &c).unwrap();

    assert!(
        tn.pauli_expectations(&[vec![PauliTerm::z(0), PauliTerm::x(0)]])
            .is_err()
    );
    assert!(tn.pauli_expectations(&[vec![PauliTerm::z(7)]]).is_err());
}

#[test]
fn par_16q_random_sparse_matches_statevector() {
    let circuit = prism_q::circuits::random_circuit(16, 10, SEED);
    let sv = run_and_probs(&circuit);
    let sp = run_sparse_probs(&circuit);
    assert_probs_par(&sp, &sv, "sparse/random_16q");
}

#[test]
fn par_16q_random_mps_matches_statevector() {
    let circuit = prism_q::circuits::random_circuit(16, 10, SEED);
    let sv = run_and_probs(&circuit);
    let mut mps = MpsBackend::new(SEED, 256);
    sim::run_on(&mut mps, &circuit).unwrap();
    let mp = mps.probabilities().unwrap();
    assert_probs_par(&mp, &sv, "mps/random_16q");
}

#[test]
fn par_16q_clifford_stabilizer_matches_statevector() {
    let circuit = prism_q::circuits::clifford_heavy_circuit(16, 10, SEED);
    let sv = run_and_probs(&circuit);
    let stab = run_stabilizer_probs(&circuit);
    assert_probs_par(&stab, &sv, "stabilizer/clifford_16q");
}

#[test]
fn par_16q_qft_sparse_matches_statevector() {
    let circuit = prism_q::circuits::qft_circuit(16);
    let sv = run_and_probs(&circuit);
    let sp = run_sparse_probs(&circuit);
    assert_probs_par(&sp, &sv, "sparse/qft_16q");
}

#[test]
fn par_16q_hea_mps_matches_statevector() {
    let circuit = prism_q::circuits::hardware_efficient_ansatz(16, 3, SEED);
    let sv = run_and_probs(&circuit);
    let mut mps = MpsBackend::new(SEED, 256);
    sim::run_on(&mut mps, &circuit).unwrap();
    let mp = mps.probabilities().unwrap();
    assert_probs_par(&mp, &sv, "mps/hea_16q");
}

#[test]
fn par_16q_product_matches_statevector() {
    let mut c = Circuit::new(16, 0);
    for q in 0..16 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::Rz(0.1 * (q + 1) as f64), &[q]);
    }
    let sv = run_and_probs(&c);
    let pp = run_product_probs(&c);
    assert_probs_par(&pp, &sv, "product/16q");
}

// ---- SX and SXdg on the stabilizer tableau ----

#[test]
fn stabilizer_sx_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::SX, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
}

#[test]
fn stabilizer_sxdg_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::SXdg, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs(&run_stabilizer_probs(&c), &run_and_probs(&c));
}

#[test]
fn stabilizer_sx_sxdg_cancel() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::SX, &[0]);
    c.add_gate(Gate::SXdg, &[0]);
    let probs = run_stabilizer_probs(&c);
    assert!((probs[0] - 1.0).abs() < EPS);
    assert!(probs[1].abs() < EPS);
}

// ---- Subsystem decomposition ----

#[test]
fn decomposed_two_independent_bell_pairs() {
    let mut c = Circuit::new(4, 0);
    // Bell pair on (0,1)
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    // Bell pair on (2,3)
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);

    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 2);

    // Decomposed via run_with (triggers decomposition)
    let decomposed = run_with(prism_q::BackendKind::Statevector, &c, SEED).unwrap();
    // Monolithic via run_on (skips decomposition)
    let mut sv = StatevectorBackend::new(SEED);
    let monolithic = sim::run_on(&mut sv, &c).unwrap();

    let dp = decomposed.probabilities.unwrap().to_vec();
    let mp = monolithic.probabilities.unwrap().to_vec();
    assert_eq!(dp.len(), mp.len());
    for (i, (d, m)) in dp.iter().zip(mp.iter()).enumerate() {
        assert!(
            (d - m).abs() < EPS,
            "prob[{i}]: decomposed={d}, monolithic={m}"
        );
    }
}

#[test]
fn decomposed_three_independent_blocks() {
    let mut c = Circuit::new(6, 0);
    // Block 0: qubits 0,1
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    // Block 1: qubits 2,3
    c.add_gate(Gate::X, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    // Block 2: qubits 4,5
    c.add_gate(Gate::H, &[4]);
    c.add_gate(Gate::Cx, &[4, 5]);

    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 3);

    let decomposed = run_with(prism_q::BackendKind::Statevector, &c, SEED).unwrap();
    let mut sv = StatevectorBackend::new(SEED);
    let monolithic = sim::run_on(&mut sv, &c).unwrap();

    let dp = decomposed.probabilities.unwrap().to_vec();
    let mp = monolithic.probabilities.unwrap().to_vec();
    for (i, (d, m)) in dp.iter().zip(mp.iter()).enumerate() {
        assert!(
            (d - m).abs() < EPS,
            "prob[{i}]: decomposed={d}, monolithic={m}"
        );
    }
}

#[test]
fn decomposed_with_measurements() {
    let mut c = Circuit::new(4, 2);
    // Block (0,1): Bell + measure
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    // Block (2,3): X + measure
    c.add_gate(Gate::X, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_measure(2, 1);

    let result = run_with(prism_q::BackendKind::Statevector, &c, SEED).unwrap();
    assert_eq!(result.classical_bits.len(), 2);
}

#[test]
fn decomposed_fully_entangled_same_as_monolithic() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Cx, &[2, 3]);

    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 1); // no decomposition

    let via_run_with = run_with(prism_q::BackendKind::Statevector, &c, SEED).unwrap();
    let mut sv = StatevectorBackend::new(SEED);
    let via_run_on = sim::run_on(&mut sv, &c).unwrap();

    let rw = via_run_with.probabilities.unwrap().to_vec();
    let ro = via_run_on.probabilities.unwrap().to_vec();
    for (i, (a, b)) in rw.iter().zip(ro.iter()).enumerate() {
        assert!((a - b).abs() < EPS, "prob[{i}]: {a} vs {b}");
    }
}

#[test]
fn decomposed_auto_mixed_backends() {
    let mut c = Circuit::new(6, 0);
    // Block (0,1,2): Clifford only, Auto should pick Stabilizer
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::S, &[0]);
    // Block (3,4,5): Non-Clifford, Auto should pick Statevector
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Cx, &[3, 4]);
    c.add_gate(Gate::Cx, &[4, 5]);

    let result = run_with(prism_q::BackendKind::Auto, &c, SEED).unwrap();
    let mut sv = StatevectorBackend::new(SEED);
    let monolithic = sim::run_on(&mut sv, &c).unwrap();

    let ap = result.probabilities.unwrap().to_vec();
    let mp = monolithic.probabilities.unwrap().to_vec();
    for (i, (a, m)) in ap.iter().zip(mp.iter()).enumerate() {
        assert!(
            (a - m).abs() < EPS,
            "prob[{i}]: auto_decomposed={a}, monolithic={m}"
        );
    }
}

// ---- Cross-backend correctness at 20q ----
//
// At 20q, all fusion passes are active (cancel, fuse_1q, reorder, fuse_2q,
// multi_1q, multi_2q, cphase, batch_post_phase) and all parallel kernels fire.

#[test]
fn par_20q_qft_sparse_matches_statevector() {
    let circuit = prism_q::circuits::qft_circuit(20);
    let sv = run_and_probs(&circuit);
    let sp = run_sparse_probs(&circuit);
    assert_probs_par(&sp, &sv, "sparse/qft_20q");
}

#[test]
fn par_20q_random_mps_matches_statevector() {
    let circuit = prism_q::circuits::random_circuit(20, 5, SEED);
    let sv = run_and_probs(&circuit);
    let mut mps = MpsBackend::new(SEED, 256);
    sim::run_on(&mut mps, &circuit).unwrap();
    let mp = mps.probabilities().unwrap();
    assert_probs_par(&mp, &sv, "mps/random_20q");
}

#[test]
fn par_20q_clifford_stabilizer_matches_statevector() {
    let circuit = prism_q::circuits::clifford_heavy_circuit(20, 10, SEED);
    let sv = run_and_probs(&circuit);
    let stab = run_stabilizer_probs(&circuit);
    assert_probs_par(&stab, &sv, "stabilizer/clifford_20q");
}

#[test]
fn par_20q_hea_factored_matches_statevector() {
    let circuit = prism_q::circuits::hardware_efficient_ansatz(20, 3, SEED);
    let sv = run_and_probs(&circuit);
    let mut fac = prism_q::backend::factored::FactoredBackend::new(SEED);
    sim::run_on(&mut fac, &circuit).unwrap();
    let fp = fac.probabilities().unwrap();
    assert_probs_par(&fp, &sv, "factored/hea_20q");
}

#[test]
fn sparse_sx_sxdg_matches_statevector() {
    let c = sx_sxdg_test_circuit();
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_sx_sxdg_matches_statevector() {
    let c = sx_sxdg_test_circuit();
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn tn_sx_sxdg_matches_statevector() {
    let c = sx_sxdg_test_circuit();
    assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
}

#[test]
fn sparse_p_theta_matches_statevector() {
    let c = p_theta_test_circuit();
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_p_theta_matches_statevector() {
    let c = p_theta_test_circuit();
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn tn_p_theta_matches_statevector() {
    let c = p_theta_test_circuit();
    assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
}

#[test]
fn sparse_cu_matches_statevector() {
    let c = cu_test_circuit();
    assert_probs(&run_sparse_probs(&c), &run_and_probs(&c));
}

#[test]
fn mps_cu_matches_statevector() {
    let c = cu_test_circuit();
    assert_probs_mps(&run_mps_probs(&c), &run_and_probs(&c));
}

#[test]
fn tn_cu_matches_statevector() {
    let c = cu_test_circuit();
    assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
}

// ---- export_statevector cross-backend tests ----

#[test]
fn sparse_export_statevector_matches() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);

    let sv_ref = run_and_state(&c);

    let mut sparse = SparseBackend::new(SEED);
    sim::run_on(&mut sparse, &c).unwrap();
    let sv = sparse.export_statevector().unwrap();
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < EPS,
            "sparse export[{i}]: expected {e}, got {a}"
        );
    }
}

#[test]
fn mps_export_statevector_matches() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);

    let sv_ref = run_and_state(&c);

    let mut mps = MpsBackend::new(SEED, 64);
    sim::run_on(&mut mps, &c).unwrap();
    let sv = mps.export_statevector().unwrap();
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < MPS_EPS,
            "mps export[{i}]: expected {e}, got {a}"
        );
    }
}

#[test]
fn tn_export_statevector_matches() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);

    let sv_ref = run_and_state(&c);

    let mut tn = TensorNetworkBackend::new(SEED);
    sim::run_on(&mut tn, &c).unwrap();
    let sv = tn.export_statevector().unwrap();
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < TN_EPS,
            "tn export[{i}]: expected {e}, got {a}"
        );
    }
}

#[test]
fn product_export_statevector_matches() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.5), &[1]);
    c.add_gate(Gate::T, &[2]);

    let sv_ref = run_and_state(&c);

    let mut prod = ProductStateBackend::new(SEED);
    sim::run_on(&mut prod, &c).unwrap();
    let sv = prod.export_statevector().unwrap();
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < EPS,
            "product export[{i}]: expected {e}, got {a}"
        );
    }
}

// One entangled block spanning every qubit, which the factored backend holds as
// a single dense sub-state, so the export is the identity case.
#[test]
fn factored_export_statevector_single_block_matches() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[1, 2]);

    let sv_ref = run_and_state(&c);

    let mut fac = prism_q::backend::factored::FactoredBackend::new(SEED);
    sim::run_on(&mut fac, &c).unwrap();
    let sv = fac.export_statevector().unwrap();
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < EPS,
            "factored export[{i}]: expected {e}, got {a}"
        );
    }
}

// Three live sub-states, one of them interleaved with another in global qubit
// order (`{0, 3}` and `{1, 2}`), so the export cannot assume a block owns a
// contiguous bit range. Amplitudes, not probabilities: a tensor product of
// blocks carries each block's phase, and only the amplitude check sees them.
#[test]
fn factored_export_statevector_interleaved_blocks_matches() {
    let mut c = Circuit::new(5, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Ry(0.9), &[4]);

    let sv_ref = run_and_state(&c);

    let mut fac = prism_q::backend::factored::FactoredBackend::new(SEED);
    sim::run_on(&mut fac, &c).unwrap();
    let sv = fac.export_statevector().unwrap();
    assert_eq!(sv.len(), sv_ref.len());
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < EPS,
            "factored export[{i}]: expected {e}, got {a}"
        );
    }
}

// After a measurement collapses one block, the surviving amplitudes must still
// come out normalized, since the export carries no pending norm of its own.
#[test]
fn factored_export_statevector_after_measurement_is_normalized() {
    let mut c = Circuit::new(4, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_measure(0, 0);

    let mut fac = prism_q::backend::factored::FactoredBackend::new(SEED);
    sim::run_on(&mut fac, &c).unwrap();
    let sv = fac.export_statevector().unwrap();

    let norm: f64 = sv.iter().map(|a| a.norm_sqr()).sum();
    assert!(
        (norm - 1.0).abs() < EPS,
        "export norm after measure: {norm}"
    );

    let probs = fac.probabilities().unwrap();
    for (i, (amp, p)) in sv.iter().zip(probs.iter()).enumerate() {
        assert!(
            (amp.norm_sqr() - p).abs() < EPS,
            "factored export[{i}] disagrees with probabilities: {} vs {p}",
            amp.norm_sqr()
        );
    }
}

// ---- MCU cross-backend tests ----

#[test]
fn tn_mcu_toffoli_matches_statevector() {
    let x_mat = Gate::X.matrix_2x2();
    let toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(toffoli, &[0, 1, 2]);
    assert_probs_tn(&run_tn_probs(&c), &run_and_probs(&c));
}

#[test]
fn factored_mcu_toffoli_matches_statevector() {
    let x_mat = Gate::X.matrix_2x2();
    let toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(toffoli, &[0, 1, 2]);

    let sv = run_and_probs(&c);
    let mut fac = prism_q::backend::factored::FactoredBackend::new(SEED);
    sim::run_on(&mut fac, &c).unwrap();
    let fp = fac.probabilities().unwrap();
    assert_probs(&fp, &sv);
}

// ---- Conditional gate cross-backend tests ----

#[test]
fn sparse_conditional_gate() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: prism_q::ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: prism_q::circuit::smallvec![1],
    });

    let sv = run_and_probs(&c);
    let sp = run_sparse_probs(&c);
    assert_probs(&sp, &sv);
}

#[test]
fn mps_conditional_gate() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: prism_q::ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: prism_q::circuit::smallvec![1],
    });

    let sv = run_and_probs(&c);
    let mp = run_mps_probs(&c);
    assert_probs_mps(&mp, &sv);
}

#[test]
fn factored_conditional_gate() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: prism_q::ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: prism_q::circuit::smallvec![1],
    });

    let sv = run_and_probs(&c);
    let mut fac = prism_q::backend::factored::FactoredBackend::new(SEED);
    sim::run_on(&mut fac, &c).unwrap();
    let fp = fac.probabilities().unwrap();
    assert_probs(&fp, &sv);
}

// Two clusters that are mutually unentangled and interleaved in register order
// (`{0, 3}` and `{1, 2}`), so the export cannot assume a cluster owns a
// contiguous bit range. Amplitudes, not probabilities: `S` on q2 puts an `i` on
// one branch that only the amplitude check sees.
#[test]
fn factored_stabilizer_export_statevector_two_clusters_matches() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::S, &[2]);

    let sv_ref = run_and_state(&c);

    let mut fs = prism_q::backend::factored_stabilizer::FactoredStabilizerBackend::new(SEED);
    sim::run_on(&mut fs, &c).unwrap();
    let sv = fs.export_statevector().unwrap();
    assert_eq!(sv.len(), sv_ref.len());
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < EPS,
            "factored-stabilizer export[{i}]: expected {e}, got {a}"
        );
    }
}

// The native marginal route (per-qubit Z expectations off the backend's own
// representation) against the dense route it replaces. The entangled fixture
// keeps every explicit kind on direct resolution; the product fixture is the one
// carried past subsystem decomposition.
#[test]
fn native_marginals_match_the_dense_route() {
    let mut entangled = Circuit::new(4, 0);
    entangled.add_gate(Gate::H, &[0]);
    entangled.add_gate(Gate::T, &[1]);
    entangled.add_gate(Gate::Cx, &[0, 1]);
    entangled.add_gate(Gate::Ry(0.7), &[2]);
    entangled.add_gate(Gate::Cx, &[1, 2]);
    entangled.add_gate(Gate::Cx, &[2, 3]);
    entangled.add_gate(Gate::Rz(0.4), &[3]);

    let mut product = Circuit::new(4, 0);
    product.add_gate(Gate::H, &[0]);
    product.add_gate(Gate::Ry(0.7), &[1]);
    product.add_gate(Gate::T, &[2]);
    product.add_gate(Gate::Rx(1.1), &[3]);

    let cases: [(prism_q::BackendKind, &Circuit, &str); 5] = [
        (prism_q::BackendKind::Sparse, &entangled, "sparse"),
        (
            prism_q::BackendKind::Mps { max_bond_dim: 64 },
            &entangled,
            "mps",
        ),
        (prism_q::BackendKind::Factored, &entangled, "factored"),
        (
            prism_q::BackendKind::DensityMatrix,
            &entangled,
            "density_matrix",
        ),
        (prism_q::BackendKind::ProductState, &product, "product"),
    ];

    for (kind, circuit, label) in cases {
        let dense = prism_q::simulate(circuit)
            .backend(prism_q::BackendKind::Statevector)
            .seed(SEED)
            .run()
            .unwrap()
            .probabilities
            .unwrap()
            .marginals();
        let native = prism_q::simulate(circuit)
            .backend(kind)
            .seed(SEED)
            .marginals()
            .unwrap()
            .into_vec();
        assert_eq!(native.len(), dense.len(), "{label}");
        for (q, (got, want)) in native.iter().zip(&dense).enumerate() {
            assert!(
                (got.0 - want.0).abs() < EPS && (got.1 - want.1).abs() < EPS,
                "{label} marginal[{q}]: got {got:?}, want {want:?}"
            );
        }
    }
}

// Explicit fused payloads pin the kernel below the fusion thresholds and in
// both target orders: iSWAP is a scaled permutation, the CZ compound is
// diagonal, the XX+YY quarter turn is dense and must stay on the general path,
// and the phased 3-cycle is asymmetric so a transposed extraction changes the
// distribution.
#[test]
fn sparse_explicit_fused_2q_shapes_sv() {
    let o = Complex64::new(0.0, 0.0);
    let l = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let iswap = [[l, o, o, o], [o, o, i, o], [o, i, o, o], [o, o, o, l]];
    let cz_s = [[l, o, o, o], [o, i, o, o], [o, o, l, o], [o, o, o, -i]];
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let c = Complex64::new(h, 0.0);
    let ic = Complex64::new(0.0, h);
    let xxyy = [[l, o, o, o], [o, c, ic, o], [o, ic, c, o], [o, o, o, l]];
    let cycle3 = [[o, o, o, -i], [o, -l, o, o], [l, o, o, o], [o, o, i, o]];

    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::Fused2q(Box::new(iswap)), &[0, 1]);
    circuit.add_gate(Gate::Fused2q(Box::new(iswap)), &[3, 2]);
    circuit.add_gate(Gate::Fused2q(Box::new(cz_s)), &[1, 2]);
    circuit.add_gate(Gate::Fused2q(Box::new(xxyy)), &[0, 3]);
    circuit.add_gate(Gate::Fused2q(Box::new(cycle3)), &[1, 3]);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }
    let mut backend = SparseBackend::new(SEED);
    common::assert_backend_matches_sv(&mut backend, &circuit, EPS, "explicit fused 2q shapes");
}
