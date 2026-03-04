//! Golden tests: known circuits with analytically verified outputs.
//!
//! Each test constructs a circuit programmatically (not via QASM) and checks
//! the resulting state vector or probabilities against hand-computed values.

use num_complex::Complex64;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::backend::Backend;
use prism_q::circuit::Circuit;
use prism_q::gates::{Gate, McuData};
use prism_q::sim;
use prism_q::Instruction;

const EPS: f64 = 1e-12;

fn run_and_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn run_and_state(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.state_vector().to_vec()
}

fn assert_probs(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < EPS, "prob[{i}]: expected {e}, got {a}");
    }
}

fn assert_amplitude(actual: Complex64, expected: Complex64, label: &str) {
    assert!(
        (actual - expected).norm() < EPS,
        "{label}: expected {expected}, got {actual}"
    );
}

// ---- Identity ----

#[test]
fn identity_preserves_zero() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Id, &[0]);
    assert_probs(&run_and_probs(&c), &[1.0, 0.0]);
}

// ---- Pauli gates ----

#[test]
fn x_flips_zero_to_one() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    assert_probs(&run_and_probs(&c), &[0.0, 1.0]);
}

#[test]
fn y_on_zero() {
    // Y|0⟩ = i|1⟩
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Y, &[0]);
    let sv = run_and_state(&c);
    assert_amplitude(sv[0], Complex64::new(0.0, 0.0), "|0⟩");
    assert_amplitude(sv[1], Complex64::new(0.0, 1.0), "|1⟩");
}

#[test]
fn z_on_zero() {
    // Z|0⟩ = |0⟩
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Z, &[0]);
    let sv = run_and_state(&c);
    assert_amplitude(sv[0], Complex64::new(1.0, 0.0), "|0⟩");
    assert_amplitude(sv[1], Complex64::new(0.0, 0.0), "|1⟩");
}

#[test]
fn z_on_plus() {
    // Z·H|0⟩ = |−⟩ = (|0⟩ − |1⟩)/√2
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Z, &[0]);
    let sv = run_and_state(&c);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert_amplitude(sv[0], Complex64::new(inv_sqrt2, 0.0), "|0⟩");
    assert_amplitude(sv[1], Complex64::new(-inv_sqrt2, 0.0), "|1⟩");
}

// ---- Hadamard ----

#[test]
fn hadamard_creates_superposition() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    assert_probs(&run_and_probs(&c), &[0.5, 0.5]);
}

#[test]
fn double_hadamard_is_identity() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[0]);
    assert_probs(&run_and_probs(&c), &[1.0, 0.0]);
}

// ---- S and T gates ----

#[test]
fn s_sdg_cancel() {
    // S·S† = I
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Sdg, &[0]);
    c.add_gate(Gate::H, &[0]);
    assert_probs(&run_and_probs(&c), &[1.0, 0.0]);
}

#[test]
fn t_tdg_cancel() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Tdg, &[0]);
    c.add_gate(Gate::H, &[0]);
    assert_probs(&run_and_probs(&c), &[1.0, 0.0]);
}

#[test]
fn s_squared_is_z() {
    // S^2 = Z. Apply H, S, S, H — should be same as H, Z, H = X, so |0⟩ → |1⟩
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::H, &[0]);
    assert_probs(&run_and_probs(&c), &[0.0, 1.0]);
}

// ---- Bell state ----

#[test]
fn bell_phi_plus() {
    // (|00⟩ + |11⟩)/√2
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs(&run_and_probs(&c), &[0.5, 0.0, 0.0, 0.5]);
}

#[test]
fn bell_psi_plus() {
    // (|01⟩ + |10⟩)/√2
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::X, &[0]);
    assert_probs(&run_and_probs(&c), &[0.0, 0.5, 0.5, 0.0]);
}

// ---- GHZ state ----

#[test]
fn ghz_4_qubit() {
    // (|0000⟩ + |1111⟩)/√2
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    let probs = run_and_probs(&c);
    assert_eq!(probs.len(), 16);
    assert!((probs[0] - 0.5).abs() < EPS);
    assert!((probs[15] - 0.5).abs() < EPS);
    let rest_sum: f64 = probs[1..15].iter().sum();
    assert!(rest_sum.abs() < EPS);
}

// ---- SWAP ----

#[test]
fn swap_exchanges_qubits() {
    // |10⟩ → SWAP → |01⟩
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    assert_probs(&run_and_probs(&c), &[0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn double_swap_is_identity() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    c.add_gate(Gate::Swap, &[0, 1]);

    let sv = run_and_state(&c);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert_amplitude(sv[0], Complex64::new(0.0, 0.0), "|00⟩");
    assert_amplitude(sv[1], Complex64::new(inv_sqrt2, 0.0), "|01⟩");
    assert_amplitude(sv[2], Complex64::new(0.0, 0.0), "|10⟩");
    assert_amplitude(sv[3], Complex64::new(inv_sqrt2, 0.0), "|11⟩");
}

// ---- CZ ----

#[test]
fn cz_on_11() {
    // Prepare |11⟩, apply CZ, check phase flip
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    let sv = run_and_state(&c);
    assert_amplitude(sv[3], Complex64::new(-1.0, 0.0), "|11⟩");
}

// ---- Rotation gates ----

#[test]
fn rx_pi_is_x_up_to_phase() {
    // Rx(π)|0⟩ = -i|1⟩
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(std::f64::consts::PI), &[0]);
    let probs = run_and_probs(&c);
    assert_probs(&probs, &[0.0, 1.0]);
}

#[test]
fn ry_pi_is_y_up_to_phase() {
    // Ry(π)|0⟩ = |1⟩
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Ry(std::f64::consts::PI), &[0]);
    let probs = run_and_probs(&c);
    assert_probs(&probs, &[0.0, 1.0]);
}

#[test]
fn rz_does_not_change_zero_probability() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rz(1.234), &[0]);
    let probs = run_and_probs(&c);
    assert_probs(&probs, &[1.0, 0.0]);
}

#[test]
fn rx_half_pi_creates_superposition() {
    // Rx(π/2)|0⟩ → equal superposition (up to phase)
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(std::f64::consts::FRAC_PI_2), &[0]);
    let probs = run_and_probs(&c);
    assert_probs(&probs, &[0.5, 0.5]);
}

// ---- Measurement ----

#[test]
fn measure_collapsed_state_is_consistent() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);

    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, &c).unwrap();
    let outcome = backend.classical_results()[0];
    let probs = backend.probabilities().unwrap();

    if outcome {
        assert!((probs[1] - 1.0).abs() < EPS);
        assert!(probs[0].abs() < EPS);
    } else {
        assert!((probs[0] - 1.0).abs() < EPS);
        assert!(probs[1].abs() < EPS);
    }
}

// ---- Circuit depth ----

#[test]
fn depth_calculation() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    // Layer 1: H(0), H(1), H(2) — all parallel
    // Layer 2: CX(0,1) — needs q0,q1 free after layer 1
    // Layer 3: CX(1,2) — needs q1 free after layer 2
    assert_eq!(c.depth(), 3);
}

// ---- Stabilizer vs Statevector golden tests ----

fn run_stabilizer_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StabilizerBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
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

// ---- Sparse vs Statevector golden tests ----

fn run_sparse_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = SparseBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
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

// ---- MPS vs Statevector golden tests ----

const MPS_EPS: f64 = 1e-10;

fn run_mps_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = MpsBackend::new(42, 64);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn assert_probs_mps(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < MPS_EPS, "prob[{i}]: expected {e}, got {a}");
    }
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

    let mut sv_backend = StatevectorBackend::new(42);
    sim::run_on(&mut sv_backend, &c).unwrap();
    let sv_bits = sv_backend.classical_results().to_vec();

    let mut mps_backend = MpsBackend::new(42, 64);
    sim::run_on(&mut mps_backend, &c).unwrap();
    let mps_bits = mps_backend.classical_results().to_vec();

    assert_eq!(sv_bits, mps_bits);
}

// ---- Multi-controlled gate (MCU) golden tests ----

#[test]
fn mcu_toffoli_both_active() {
    // Toffoli (CCX): |110⟩ → |111⟩
    let x_mat = Gate::X.matrix_2x2();
    let toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(toffoli, &[1, 2, 0]);
    assert_probs(
        &run_and_probs(&c),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    );
}

#[test]
fn mcu_toffoli_one_inactive() {
    // Toffoli (CCX): |100⟩ → |100⟩ (only 1 control active, no flip)
    let x_mat = Gate::X.matrix_2x2();
    let toffoli = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(toffoli, &[1, 2, 0]);
    assert_probs(
        &run_and_probs(&c),
        &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
}

#[test]
fn mcu_ccz_phase_flip() {
    // CCZ: phase-flip |111⟩. Prepare equal superposition, check |111⟩ gets -1.
    let z_mat = Gate::Z.matrix_2x2();
    let ccz = Gate::Mcu(Box::new(McuData {
        mat: z_mat,
        num_controls: 2,
    }));
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(ccz, &[0, 1, 2]);
    let sv = run_and_state(&c);
    let amp = 1.0 / 8.0_f64.sqrt();
    for (i, a) in sv.iter().enumerate() {
        if i == 7 {
            assert_amplitude(*a, Complex64::new(-amp, 0.0), "|111⟩");
        } else {
            assert_amplitude(*a, Complex64::new(amp, 0.0), &format!("|{i:03b}⟩"));
        }
    }
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

#[test]
fn mcu_3ctrl_x() {
    // CCCX: 3 controls, flip target only when all active
    let x_mat = Gate::X.matrix_2x2();
    let cccx = Gate::Mcu(Box::new(McuData {
        mat: x_mat,
        num_controls: 3,
    }));
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(cccx, &[0, 1, 2, 3]);
    let probs = run_and_probs(&c);
    assert!(
        (probs[0b1111] - 1.0).abs() < EPS,
        "all controls active should flip target"
    );
}

#[test]
fn mcu_inv_ctrl_ctrl_rz() {
    // inv @ ctrl @ ctrl @ rz(pi/4): apply CRz(−π/4) with 2 controls
    // Set all controls active, put target in superposition
    let rz_mat = Gate::Rz(std::f64::consts::FRAC_PI_4).matrix_2x2();
    let rz_inv_mat = Gate::Rz(-std::f64::consts::FRAC_PI_4).matrix_2x2();
    let mcu_fwd = Gate::Mcu(Box::new(McuData {
        mat: rz_mat,
        num_controls: 2,
    }));
    let mcu_inv = Gate::Mcu(Box::new(McuData {
        mat: rz_inv_mat,
        num_controls: 2,
    }));

    // Apply forward then inverse — should cancel to identity on target
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(mcu_fwd, &[0, 1, 2]);
    c.add_gate(mcu_inv, &[0, 1, 2]);
    let sv = run_and_state(&c);
    // State should be |q0=1,q1=1⟩|+⟩ = (|011⟩ + |111⟩)/√2
    let amp = 1.0 / 2.0_f64.sqrt();
    assert_amplitude(sv[0b011], Complex64::new(amp, 0.0), "|011⟩");
    assert_amplitude(sv[0b111], Complex64::new(amp, 0.0), "|111⟩");
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

// ---- Product state backend ----

fn run_product_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = ProductStateBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
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
fn product_rejects_entangling() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = ProductStateBackend::new(42);
    let result = sim::run_on(&mut b, &c);
    assert!(result.is_err());
}

// ---- Tensor Network vs Statevector golden tests ----

const TN_EPS: f64 = 1e-10;

fn run_tn_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = TensorNetworkBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn assert_probs_tn(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < TN_EPS, "prob[{i}]: expected {e}, got {a}");
    }
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

// ---- Parallel threshold correctness (16q = PARALLEL_THRESHOLD_QUBITS + 2) ----
//
// Statevector switches to Rayon parallel kernels at >= 14 qubits.
// These tests run at 16q to validate parallel paths produce identical
// results to sequential backends.

const PAR_EPS: f64 = 1e-10;

fn assert_probs_par(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < PAR_EPS,
            "{label} prob[{i}]: expected {e}, got {a}"
        );
    }
}

#[test]
fn par_16q_random_sparse_matches_statevector() {
    let circuit = prism_q::circuits::random_circuit(16, 10, 42);
    let sv = run_and_probs(&circuit);
    let sp = run_sparse_probs(&circuit);
    assert_probs_par(&sp, &sv, "sparse/random_16q");
}

#[test]
fn par_16q_random_mps_matches_statevector() {
    let circuit = prism_q::circuits::random_circuit(16, 10, 42);
    let sv = run_and_probs(&circuit);
    let mut mps = MpsBackend::new(42, 256);
    sim::run_on(&mut mps, &circuit).unwrap();
    let mp = mps.probabilities().unwrap();
    assert_probs_par(&mp, &sv, "mps/random_16q");
}

#[test]
fn par_16q_clifford_stabilizer_matches_statevector() {
    let circuit = prism_q::circuits::clifford_heavy_circuit(16, 10, 42);
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
    let circuit = prism_q::circuits::hardware_efficient_ansatz(16, 3, 42);
    let sv = run_and_probs(&circuit);
    let mut mps = MpsBackend::new(42, 256);
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

// ---- New gate golden tests ----

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

#[test]
fn crx_matches_ctrl_rx() {
    let qasm_crx = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncrx(pi/3) q[0], q[1];";
    let qasm_ctrl = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nctrl @ rx(pi/3) q[0], q[1];";
    let mut b1 = StatevectorBackend::new(42);
    let mut b2 = StatevectorBackend::new(42);
    let c1 = prism_q::circuit::openqasm::parse(qasm_crx).unwrap();
    let c2 = prism_q::circuit::openqasm::parse(qasm_ctrl).unwrap();
    sim::run_on(&mut b1, &c1).unwrap();
    sim::run_on(&mut b2, &c2).unwrap();
    assert_probs(&b1.probabilities().unwrap(), &b2.probabilities().unwrap());
}

#[test]
fn ccx_matches_ctrl_ctrl_x() {
    let qasm_ccx = "OPENQASM 3.0;\nqubit[3] q;\nh q[0];\nh q[1];\nccx q[0], q[1], q[2];";
    let qasm_ctrl =
        "OPENQASM 3.0;\nqubit[3] q;\nh q[0];\nh q[1];\nctrl @ ctrl @ x q[0], q[1], q[2];";
    let mut b1 = StatevectorBackend::new(42);
    let mut b2 = StatevectorBackend::new(42);
    let c1 = prism_q::circuit::openqasm::parse(qasm_ccx).unwrap();
    let c2 = prism_q::circuit::openqasm::parse(qasm_ctrl).unwrap();
    sim::run_on(&mut b1, &c1).unwrap();
    sim::run_on(&mut b2, &c2).unwrap();
    assert_probs(&b1.probabilities().unwrap(), &b2.probabilities().unwrap());
}

#[test]
fn rzz_decomposition_correct() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nh q[1];\nrzz(pi/4) q[0], q[1];";
    let mut b = StatevectorBackend::new(42);
    let c = prism_q::circuit::openqasm::parse(qasm).unwrap();
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    // After H|0>H|0> = |++>, then Rzz(pi/4):
    // Manual: CX + Rz(pi/4) + CX on |++>
    let mut c2 = Circuit::new(2, 0);
    c2.add_gate(Gate::H, &[0]);
    c2.add_gate(Gate::H, &[1]);
    c2.add_gate(Gate::Cx, &[0, 1]);
    c2.add_gate(Gate::Rz(std::f64::consts::FRAC_PI_4), &[1]);
    c2.add_gate(Gate::Cx, &[0, 1]);
    assert_probs(&probs, &run_and_probs(&c2));
}

#[test]
fn u3_matches_manual() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nu3(pi/2, 0, pi) q[0];";
    let mut b = StatevectorBackend::new(42);
    let c = prism_q::circuit::openqasm::parse(qasm).unwrap();
    sim::run_on(&mut b, &c).unwrap();
    // u3(pi/2, 0, pi) = H up to global phase
    let probs = b.probabilities().unwrap();
    let h_probs = run_and_probs(&{
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c
    });
    assert_probs(&probs, &h_probs);
}

#[test]
fn cswap_correct() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nx q[2];\ncswap q[0], q[1], q[2];";
    let mut b = StatevectorBackend::new(42);
    let c = prism_q::circuit::openqasm::parse(qasm).unwrap();
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    // ctrl=|0>, so swap doesn't activate: result = |001> (bit 2 flipped = index 4)
    assert!((probs[4] - 1.0).abs() < EPS);

    // With ctrl=|1>: should swap q[1] and q[2]
    let qasm2 = "OPENQASM 3.0;\nqubit[3] q;\nx q[0];\nx q[2];\ncswap q[0], q[1], q[2];";
    let mut b2 = StatevectorBackend::new(42);
    let c2 = prism_q::circuit::openqasm::parse(qasm2).unwrap();
    sim::run_on(&mut b2, &c2).unwrap();
    let probs2 = b2.probabilities().unwrap();
    // x q[0] -> |001>, x q[2] -> |101>, cswap swaps q[1],q[2] -> |011> = index 3 (q0=1, q1=1, q2=0)
    assert!((probs2[3] - 1.0).abs() < EPS);
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
    let decomposed = sim::run_with(prism_q::BackendKind::Statevector, &c, 42).unwrap();
    // Monolithic via run_on (skips decomposition)
    let mut sv = StatevectorBackend::new(42);
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

    let decomposed = sim::run_with(prism_q::BackendKind::Statevector, &c, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
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

    let result = sim::run_with(prism_q::BackendKind::Statevector, &c, 42).unwrap();
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

    let via_run_with = sim::run_with(prism_q::BackendKind::Statevector, &c, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
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
    // Block (0,1,2): Clifford only — Auto should pick Stabilizer
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::S, &[0]);
    // Block (3,4,5): Non-Clifford — Auto should pick Statevector
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Cx, &[3, 4]);
    c.add_gate(Gate::Cx, &[4, 5]);

    let result = sim::run_with(prism_q::BackendKind::Auto, &c, 42).unwrap();
    let mut sv = StatevectorBackend::new(42);
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
// These tests cross-validate parallel statevector against independent backends.

#[test]
fn par_20q_qft_sparse_matches_statevector() {
    let circuit = prism_q::circuits::qft_circuit(20);
    let sv = run_and_probs(&circuit);
    let sp = run_sparse_probs(&circuit);
    assert_probs_par(&sp, &sv, "sparse/qft_20q");
}

#[test]
fn par_20q_random_mps_matches_statevector() {
    let circuit = prism_q::circuits::random_circuit(20, 5, 42);
    let sv = run_and_probs(&circuit);
    let mut mps = MpsBackend::new(42, 256);
    sim::run_on(&mut mps, &circuit).unwrap();
    let mp = mps.probabilities().unwrap();
    assert_probs_par(&mp, &sv, "mps/random_20q");
}

#[test]
fn par_20q_clifford_stabilizer_matches_statevector() {
    let circuit = prism_q::circuits::clifford_heavy_circuit(20, 10, 42);
    let sv = run_and_probs(&circuit);
    let stab = run_stabilizer_probs(&circuit);
    assert_probs_par(&stab, &sv, "stabilizer/clifford_20q");
}

#[test]
fn par_20q_hea_factored_matches_statevector() {
    let circuit = prism_q::circuits::hardware_efficient_ansatz(20, 3, 42);
    let sv = run_and_probs(&circuit);
    let mut fac = prism_q::backend::factored::FactoredBackend::new(42);
    sim::run_on(&mut fac, &circuit).unwrap();
    let fp = fac.probabilities().unwrap();
    assert_probs_par(&fp, &sv, "factored/hea_20q");
}

// ---- Cross-backend SX/SXdg tests ----

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

// ---- Cross-backend P(theta) tests ----

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

// ---- Cross-backend Cu tests ----

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

    let mut sparse = SparseBackend::new(42);
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

    let mut mps = MpsBackend::new(42, 64);
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

    let mut tn = TensorNetworkBackend::new(42);
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

    let mut prod = ProductStateBackend::new(42);
    sim::run_on(&mut prod, &c).unwrap();
    let sv = prod.export_statevector().unwrap();
    for (i, (a, e)) in sv.iter().zip(sv_ref.iter()).enumerate() {
        assert!(
            (a - e).norm() < EPS,
            "product export[{i}]: expected {e}, got {a}"
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
    let mut fac = prism_q::backend::factored::FactoredBackend::new(42);
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
    let mut fac = prism_q::backend::factored::FactoredBackend::new(42);
    sim::run_on(&mut fac, &c).unwrap();
    let fp = fac.probabilities().unwrap();
    assert_probs(&fp, &sv);
}
