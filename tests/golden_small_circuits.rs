//! Closed-form goldens: circuits whose output was computed by hand. The
//! expected value is a literal from the algebra in the comment above it, or a
//! textbook decomposition of the same gate on the same backend, never a
//! second simulator: a comparison can only report that two implementations
//! disagree, not which one is right. Behavior needing an authority goes here;
//! agreement between implementations goes in `tests/backend_equivalence.rs`.

mod common;

use common::{assert_probs_close, run_fused_probs};
use num_complex::Complex64;
use prism_q::CircuitBuilder;
use prism_q::Instruction;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::Circuit;
use prism_q::gates::{Gate, McuData};
use prism_q::sim;

const EPS: f64 = 1e-12;

fn run_and_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut StatevectorBackend::new(common::SEED), circuit)
}

fn run_and_state(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(common::SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.state_vector().to_vec()
}

fn run_stabilizer_probs(circuit: &Circuit) -> Vec<f64> {
    run_fused_probs(&mut StabilizerBackend::new(common::SEED), circuit)
}

fn assert_probs(actual: &[f64], expected: &[f64]) {
    assert_probs_close(actual, expected, EPS, "golden");
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
    // S^2 = Z. Apply H, S, S, H, should be same as H, Z, H = X, so |0⟩ → |1⟩
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
    // CZ|11⟩ = -|11⟩
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

    let mut backend = StatevectorBackend::new(common::SEED);
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

// ---- Reset ----
//
// These anchor the statevector's own `reset` against hand-computed values.
// The cross-backend matrices in `tests/measurement_matrix.rs` compare every
// other backend against the statevector, so nothing there can tell which side
// of a disagreement is wrong. `reset` went unanchored here for a long time,
// and a projection-onto-|0> implementation survived as a result.

#[test]
fn reset_returns_qubit_to_zero() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_reset(0);
    assert_probs(&run_and_probs(&c), &[1.0, 0.0]);
}

#[test]
fn reset_from_one_preserves_a_spectator_superposition() {
    // Resetting a qubit that holds |1> must clear that qubit and nothing else.
    // q0 stays in |+>, so the two |q1 = 0> outcomes keep weight 1/2 each.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::H, &[0]);
    c.add_reset(1);
    assert_probs(&run_and_probs(&c), &[0.5, 0.5, 0.0, 0.0]);
}

#[test]
fn reset_of_an_entangled_partner_samples_both_branches() {
    // Reset is the channel `rho -> |0><0| (x) tr_q rho`. A statevector carries
    // one trajectory of it, so each run lands wholly on |00> or wholly on |01>
    // (q0 set, q1 cleared), and over seeds both branches must occur. A
    // projection onto |0> yields |00> every time, keeping the partner
    // correlated with the |0> outcome. Sixty-four seeds make a one-sided
    // result conclusive; `tests/reset_channel.rs` pins the 1/2 weights against
    // the density-matrix oracle.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_reset(1);

    let mut branches = [false; 2];
    for seed in 0..64u64 {
        let mut backend = StatevectorBackend::new(seed);
        sim::run_on(&mut backend, &c).unwrap();
        let probs = backend.probabilities().unwrap();
        assert!(
            probs[2].abs() < EPS && probs[3].abs() < EPS,
            "seed {seed}: q1 must be cleared, got {probs:?}"
        );
        let branch = usize::from(probs[1] > 0.5);
        assert!(
            (probs[branch] - 1.0).abs() < EPS,
            "seed {seed}: one trajectory is a single basis state, got {probs:?}"
        );
        branches[branch] = true;
    }
    assert!(
        branches[0] && branches[1],
        "reset must sample both branches of the partner, saw only {}",
        if branches[0] { "|00>" } else { "|01>" }
    );
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
    // Layer 1: H(0), H(1), H(2), all parallel
    // Layer 2: CX(0,1), needs q0,q1 free after layer 1
    // Layer 3: CX(1,2), needs q1 free after layer 2
    assert_eq!(c.depth(), 3);
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
    // CCZ: phase-flip |111⟩
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

#[test]
fn mcu_10ctrl_x_builder() {
    let mut builder = CircuitBuilder::new(11);
    for q in 0..10 {
        builder.x(q);
    }
    builder.mcu(Gate::X.matrix_2x2(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10);
    let probs = run_and_probs(&builder.build());
    assert!(
        (probs[0b111_1111_1111] - 1.0).abs() < EPS,
        "10 active controls should flip the target"
    );
}

#[test]
fn mcu_10ctrl_x_openqasm_chain() {
    let mut qasm = String::from("OPENQASM 3.0;\nqubit[11] q;\n");
    for q in 0..10 {
        qasm.push_str(&format!("x q[{q}];\n"));
    }
    qasm.push_str(&"ctrl @ ".repeat(10));
    qasm.push_str("x q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];\n");
    let c = prism_q::circuit::openqasm::parse(&qasm).unwrap();
    let probs = run_and_probs(&c);
    assert!(
        (probs[0b111_1111_1111] - 1.0).abs() < EPS,
        "10 active controls should flip the target"
    );
}

// ---- Product state backend ----

#[test]
fn product_rejects_entangling() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = ProductStateBackend::new(common::SEED);
    let result = sim::run_on(&mut b, &c);
    assert!(result.is_err());
}

// ---- Compound gates against their textbook decomposition ----
//
// The reference here is a decomposition written out by hand, applied on the
// same backend. That makes these gate identities, not a second opinion from a
// second simulator.

#[test]
fn crx_matches_ctrl_rx() {
    let qasm_crx = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncrx(pi/3) q[0], q[1];";
    let qasm_ctrl = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nctrl @ rx(pi/3) q[0], q[1];";
    let mut b1 = StatevectorBackend::new(common::SEED);
    let mut b2 = StatevectorBackend::new(common::SEED);
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
    let mut b1 = StatevectorBackend::new(common::SEED);
    let mut b2 = StatevectorBackend::new(common::SEED);
    let c1 = prism_q::circuit::openqasm::parse(qasm_ccx).unwrap();
    let c2 = prism_q::circuit::openqasm::parse(qasm_ctrl).unwrap();
    sim::run_on(&mut b1, &c1).unwrap();
    sim::run_on(&mut b2, &c2).unwrap();
    assert_probs(&b1.probabilities().unwrap(), &b2.probabilities().unwrap());
}

#[test]
fn rzz_decomposition_correct() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nh q[1];\nrzz(pi/4) q[0], q[1];";
    let mut b = StatevectorBackend::new(common::SEED);
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
    let mut b = StatevectorBackend::new(common::SEED);
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
    let mut b = StatevectorBackend::new(common::SEED);
    let c = prism_q::circuit::openqasm::parse(qasm).unwrap();
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    // ctrl=|0>, so swap doesn't activate: result = |001> (bit 2 flipped = index 4)
    assert!((probs[4] - 1.0).abs() < EPS);

    // With ctrl=|1>: should swap q[1] and q[2]
    let qasm2 = "OPENQASM 3.0;\nqubit[3] q;\nx q[0];\nx q[2];\ncswap q[0], q[1], q[2];";
    let mut b2 = StatevectorBackend::new(common::SEED);
    let c2 = prism_q::circuit::openqasm::parse(qasm2).unwrap();
    sim::run_on(&mut b2, &c2).unwrap();
    let probs2 = b2.probabilities().unwrap();
    // x q[0] -> |001>, x q[2] -> |101>, cswap swaps q[1],q[2] -> |011> = index 3 (q0=1, q1=1, q2=0)
    assert!((probs2[3] - 1.0).abs() < EPS);
}

// ---- Closed-form anchors for the conformance harness ----
//
// `tests/conformance_matrix.rs` decides disagreements by consensus, then by the
// stabilizer tableau and the density-matrix channel. Those two are the harness's
// independent authorities, so each needs values that were computed by hand
// rather than read off another backend. `Family::golden_anchor` in
// `tests/common/conformance.rs` names the test that covers each motif.

#[test]
fn stabilizer_bell_between_hadamard_layers_matches_closed_form() {
    // H q0, CX q0 q1 gives (|00> + |11>)/sqrt(2). Then H on both qubits sends
    // |00> to (|00> + |01> + |10> + |11>)/2 and |11> to
    // (|00> - |01> - |10> + |11>)/2; the cross terms cancel and the state is
    // (|00> + |11>)/sqrt(2) again.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    assert_probs(&run_stabilizer_probs(&c), &[0.5, 0.0, 0.0, 0.5]);
}

#[test]
fn density_matrix_reset_of_an_entangled_partner_is_the_channel() {
    // Reset is `rho -> |0><0| (x) tr_q rho`. Tracing qubit 1 out of a Bell pair
    // leaves qubit 0 maximally mixed, so the weight splits evenly between |00>
    // and |01> (q0 set, q1 clear). A projection onto |0> would leave all the
    // weight on |00>.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_reset(1);

    let mut backend = DensityMatrixBackend::new(common::SEED);
    sim::run_on(&mut backend, &c).unwrap();
    assert_probs(&backend.probabilities().unwrap(), &[0.5, 0.5, 0.0, 0.0]);
}

#[test]
fn density_matrix_dephasing_removes_coherence() {
    // The conformance harness lowers `measure` to the Kraus pair
    // {|0><0|, |1><1|} to build the exact unconditional mixture. That pair must
    // behave as the measurement channel, not as the identity: applying it
    // between two Hadamards has to leave |+> maximally mixed, so the second
    // Hadamard returns 1/2 on each outcome instead of returning to |0>.
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let dephasing = [[[one, zero], [zero, zero]], [[zero, zero], [zero, one]]];
    let hadamard = Instruction::Gate {
        gate: Gate::H,
        targets: prism_q::circuit::smallvec![0],
    };

    let mut backend = DensityMatrixBackend::new(common::SEED);
    backend.init(1, 0).unwrap();
    backend.apply(&hadamard).unwrap();
    backend.apply_1q_kraus(0, &dephasing);
    backend.apply(&hadamard).unwrap();
    assert_probs(&backend.probabilities().unwrap(), &[0.5, 0.5]);

    // On half of a Bell pair the same channel kills the coherence between |00>
    // and |11> while leaving both populations at 1/2, and a Hadamard on the
    // dephased qubit then spreads the mixture evenly over all four outcomes.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);

    let mut backend = DensityMatrixBackend::new(common::SEED);
    sim::run_on(&mut backend, &c).unwrap();
    backend.apply_1q_kraus(0, &dephasing);
    assert_probs(&backend.probabilities().unwrap(), &[0.5, 0.0, 0.0, 0.5]);
    backend.apply(&hadamard).unwrap();
    assert_probs(&backend.probabilities().unwrap(), &[0.25, 0.25, 0.25, 0.25]);
}

// ---- Start states ----

// H (cos(pi/8)|0> + sin(pi/8)|1>) = ((c+s)|0> + (c-s)|1>)/sqrt(2), so
// p(0) = (c+s)^2/2 = (1 + 2cs)/2 = (1 + sin(pi/4))/2 and p(1) is its
// complement. The density matrix holds |psi><psi| for the same start state,
// and its diagonal is the same pair.
#[test]
fn hadamard_on_a_tilted_start_state() {
    let theta = std::f64::consts::FRAC_PI_8;
    let state = vec![
        Complex64::new(theta.cos(), 0.0),
        Complex64::new(theta.sin(), 0.0),
    ];
    let a = std::f64::consts::FRAC_1_SQRT_2;
    let expected = [(1.0 + a) / 2.0, (1.0 - a) / 2.0];

    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);

    let mut sv = StatevectorBackend::new(common::SEED);
    sv.init_from_amplitudes(state.clone(), 0).unwrap();
    sv.apply_instructions(&c.instructions).unwrap();
    assert_probs(&sv.probabilities().unwrap(), &expected);

    let mut dm = DensityMatrixBackend::new(common::SEED);
    dm.init_from_amplitudes(state, 0).unwrap();
    dm.apply_instructions(&c.instructions).unwrap();
    assert_probs(&dm.probabilities().unwrap(), &expected);
}

// CX with the control in cos(pi/8)|0> + sin(pi/8)|1> and the target in |0>
// gives c|00> + s|11>, so the populations are cos^2(pi/8) and sin^2(pi/8) on
// the two correlated outcomes and zero elsewhere.
#[test]
fn cx_on_a_tilted_control_start_state() {
    let theta = std::f64::consts::FRAC_PI_8;
    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    state[0] = Complex64::new(theta.cos(), 0.0);
    state[1] = Complex64::new(theta.sin(), 0.0);
    let a = std::f64::consts::FRAC_1_SQRT_2;

    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);

    let mut sv = StatevectorBackend::new(common::SEED);
    sv.init_from_amplitudes(state, 0).unwrap();
    sv.apply_instructions(&c.instructions).unwrap();
    assert_probs(
        &sv.probabilities().unwrap(),
        &[(1.0 + a) / 2.0, 0.0, 0.0, (1.0 - a) / 2.0],
    );
}
