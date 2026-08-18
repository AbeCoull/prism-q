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

// A two-qubit Kraus set is indexed `K[t][t']` with `t = 2*bit(q0) + bit(q1)`,
// so the first qubit named is the high bit. `diag(1, 1, -1, -1)` is therefore
// `Z` on the first and identity on the second, and swapping the two qubit
// arguments has to move the dephasing to the other qubit.
#[test]
fn density_matrix_two_qubit_kraus_orders_the_first_qubit_high() {
    let p = 0.25f64;
    let zero = Complex64::new(0.0, 0.0);
    let diag = |scale: f64, signs: [f64; 4]| {
        let mut m = [[zero; 4]; 4];
        for (t, sign) in signs.iter().enumerate() {
            m[t][t] = Complex64::new(scale * sign, 0.0);
        }
        m
    };
    // K0 = sqrt(1-p) I, K1 = sqrt(p) (Z (x) I): dephasing of strength p on the
    // high qubit alone. On |++> that leaves <X> = 1 - 2p there and <X> = 1 on
    // the untouched one.
    let z_on_high = [
        diag((1.0 - p).sqrt(), [1.0, 1.0, 1.0, 1.0]),
        diag(p.sqrt(), [1.0, 1.0, -1.0, -1.0]),
    ];

    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);

    for (q0, q1, x0, x1) in [(0, 1, 1.0 - 2.0 * p, 1.0), (1, 0, 1.0, 1.0 - 2.0 * p)] {
        let mut backend = DensityMatrixBackend::new(common::SEED);
        sim::run_on(&mut backend, &c).unwrap();
        backend.apply_2q_kraus(q0, q1, &z_on_high);
        let got = backend
            .pauli_expectations(&[
                vec![prism_q::PauliTerm::x(0)],
                vec![prism_q::PauliTerm::x(1)],
            ])
            .unwrap();
        assert!(
            (got[0] - x0).abs() < EPS && (got[1] - x1).abs() < EPS,
            "({q0},{q1}): expected ({x0}, {x1}), got ({}, {})",
            got[0],
            got[1]
        );
    }
}

// Correlated dephasing {sqrt(1-p) I, sqrt(p) Z(x)Z} maps |++> onto the mixture
// (1-p)|++><++| + p|--><--|. Both single-qubit <X> read 1 - 2p, while <X0 X1>
// stays at 1 because the two branches flip together. Independent single-qubit
// dephasing at the same rate would give (1-2p)^2 there, so the joint term is
// what separates a correlated channel from two local ones.
#[test]
fn density_matrix_correlated_zz_dephasing_keeps_the_joint_term() {
    let p = 0.25f64;
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);

    let mut backend = DensityMatrixBackend::new(common::SEED);
    sim::run_on(&mut backend, &c).unwrap();
    let zero = Complex64::new(0.0, 0.0);
    let diag = |scale: f64, signs: [f64; 4]| {
        let mut m = [[zero; 4]; 4];
        for (t, sign) in signs.iter().enumerate() {
            m[t][t] = Complex64::new(scale * sign, 0.0);
        }
        m
    };
    let zz = [
        diag((1.0f64 - p).sqrt(), [1.0, 1.0, 1.0, 1.0]),
        diag(p.sqrt(), [1.0, -1.0, -1.0, 1.0]),
    ];
    backend.apply_2q_kraus(0, 1, &zz);

    let got = backend
        .pauli_expectations(&[
            vec![prism_q::PauliTerm::x(0)],
            vec![prism_q::PauliTerm::x(1)],
            vec![prism_q::PauliTerm::x(0), prism_q::PauliTerm::x(1)],
        ])
        .unwrap();
    let expected = [1.0 - 2.0 * p, 1.0 - 2.0 * p, 1.0];
    for (i, (a, e)) in got.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < EPS, "term {i}: expected {e}, got {a}");
    }
}

// `reduced_density_matrix_2q` packs `t = 2*bit(q0) + bit(q1)`. On
// (|00> + |01>)/sqrt(2), written with qubit 0 in the low state-index bit, the
// occupied pair is t in {0, 2} when qubit 0 is named first and t in {0, 1}
// when qubit 1 is, with every entry of the block equal to 1/2.
#[test]
fn statevector_two_qubit_reduced_density_matrix_orders_the_first_qubit_high() {
    use prism_q::backend::Backend;
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);

    let mut backend = StatevectorBackend::new(common::SEED);
    sim::run_on(&mut backend, &c).unwrap();

    for (q0, q1, occupied) in [(0usize, 1usize, [0usize, 2usize]), (1, 0, [0, 1])] {
        let rho = backend.reduced_density_matrix_2q(q0, q1).unwrap();
        for (t, row) in rho.iter().enumerate() {
            for (tp, entry) in row.iter().enumerate() {
                let expected = if occupied.contains(&t) && occupied.contains(&tp) {
                    Complex64::new(0.5, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert_amplitude(*entry, expected, &format!("({q0},{q1}) rho[{t}][{tp}]"));
            }
        }
    }
}

// `Gate::Fused2q` indexes its matrix `i*2 + j` with `i` on `targets[0]`, and the
// two-qubit Kraus path relies on that packing. `diag(1, 1, -1, -1)` is `Z` on
// the high index, so on `|++>` it takes the first-named target to `|->` and
// leaves the other at `|+>`. A closing Hadamard on both turns that into the one
// basis state with the first-named target set. Nothing else on the CPU path
// pins this: the fused fixtures elsewhere are symmetric or are compared against
// another `Fused2q`.
#[test]
fn fused_2q_orders_the_first_target_high() {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let neg = Complex64::new(-1.0, 0.0);
    let z_on_high = [
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, neg, zero],
        [zero, zero, zero, neg],
    ];

    // State index bit 0 is qubit 0, so naming qubit 0 first lands on index 1
    // and naming qubit 1 first lands on index 2.
    for (q0, q1, expected) in [
        (0usize, 1usize, [0.0, 1.0, 0.0, 0.0]),
        (1, 0, [0.0, 0.0, 1.0, 0.0]),
    ] {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::Fused2q(Box::new(z_on_high)), &[q0, q1]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        assert_probs(&run_and_probs(&c), &expected);
    }
}

// A complex, non-Hermitian two-qubit Kraus operator. Every other two-qubit set
// in the suite is real, and the Pauli depolarizing channel is invariant under
// conjugating all its operators, so a conjugate swap in the superoperator sum
// would survive them. `K = sqrt(p) |00><11|` moves the `|11>` population onto
// `|00>` with an `i` phase that only the `K rho K^dagger` order reproduces.
#[test]
fn density_matrix_two_qubit_kraus_uses_the_conjugate_transpose() {
    let p = 0.4f64;
    let zero = Complex64::new(0.0, 0.0);
    let mut keep = [[zero; 4]; 4];
    for (t, row) in keep.iter_mut().enumerate() {
        row[t] = Complex64::new(if t == 3 { (1.0 - p).sqrt() } else { 1.0 }, 0.0);
    }
    let mut jump = [[zero; 4]; 4];
    jump[0][3] = Complex64::new(0.0, p.sqrt());

    // Start from (|00> + |11>)/sqrt(2), which has coherence between the two
    // levels the jump connects.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);

    let mut backend = DensityMatrixBackend::new(common::SEED);
    sim::run_on(&mut backend, &c).unwrap();
    backend.apply_2q_kraus(0, 1, &[keep, jump]);

    // rho = 1/2 (|00><00| + |11><11|) with coherence sqrt(1-p)/2, plus the
    // jump branch p/2 landing back on |00>. Populations: |00> = (1 + p)/2,
    // |11> = (1 - p)/2. The jump term is |00><00| whatever the phase, so the
    // diagonal alone does not test the conjugate; <X0 X1> does, and it reads
    // sqrt(1-p) with the correct order.
    let probs = backend.probabilities().unwrap();
    assert_probs(&probs, &[(1.0 + p) / 2.0, 0.0, 0.0, (1.0 - p) / 2.0]);

    let got = backend
        .pauli_expectations(&[vec![prism_q::PauliTerm::x(0), prism_q::PauliTerm::x(1)]])
        .unwrap();
    assert!(
        (got[0] - (1.0 - p).sqrt()).abs() < EPS,
        "expected <X0 X1> = {}, got {}",
        (1.0 - p).sqrt(),
        got[0]
    );
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

// ===== Weighted-observable variance anchors, forced onto the grouped
// statevector route so the Clifford engines cannot serve them =====

// |+>: <Z>=0 with <Z^2>=1 gives Var(2Z0)=4; <X>=1 gives Var(X0)=0. Two
// groups (X and Z collide on qubit 0), mean 2*0 + 1 = 1, variance 4 + 0 = 4.
#[test]
fn weighted_observable_variance_on_plus_is_four() {
    use prism_q::{BackendKind, PauliObservable, PauliTerm};

    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let observable =
        PauliObservable::from_terms([(2.0, vec![PauliTerm::z(0)]), (1.0, vec![PauliTerm::x(0)])])
            .unwrap();
    let result = sim::simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(common::SEED)
        .observable_expectation(&observable)
        .unwrap();
    assert!((result.mean - 1.0).abs() < 1e-12);
    let variances = result.group_variances.unwrap();
    assert_eq!(variances.len(), 2);
    assert!((result.variance.unwrap() - 4.0).abs() < 1e-12);
}

// H = X0 + Y1 on |00> is one qubit-wise-commuting group exercising both the
// H and the Sdg-H rotations at once: mean 0, H^2 = 2I + 2*X0Y1 with
// <X0 Y1> = 0, so Var(H) = 2 exactly (single group, no excluded covariance).
#[test]
fn weighted_observable_single_group_variance_is_two() {
    use prism_q::{BackendKind, PauliObservable, PauliTerm};

    let c = Circuit::new(2, 0);
    let observable =
        PauliObservable::from_terms([(1.0, vec![PauliTerm::x(0)]), (1.0, vec![PauliTerm::y(1)])])
            .unwrap();
    assert_eq!(observable.num_groups(), 1);
    let result = sim::simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(common::SEED)
        .observable_expectation(&observable)
        .unwrap();
    assert!(result.mean.abs() < 1e-12);
    assert!((result.variance.unwrap() - 2.0).abs() < 1e-12);
}

// S H |0> = (|0> + i|1>)/sqrt(2) is the +1 eigenstate of Y: Y|0> = i|1> and
// Y|1> = -i|0>, so Y(|0> + i|1>) = |0> + i|1>. A sign error in the i^y phase
// of the term evaluation would report -1.
#[test]
fn weighted_observable_y_sign_is_pinned() {
    use prism_q::{BackendKind, PauliObservable, PauliTerm};

    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    let observable = PauliObservable::from_terms([(1.0, vec![PauliTerm::y(0)])]).unwrap();
    let result = sim::simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(common::SEED)
        .observable_expectation(&observable)
        .unwrap();
    assert!((result.mean - 1.0).abs() < 1e-12);
    assert!(result.variance.unwrap().abs() < 1e-12);
}

// Twelve disjoint X factors qubit-wise commute, so H = sum X_i is one group
// of 66 pairs, past the pair budget: the basis-rotated moments pass serves
// it. On |0..0>: <X_i> = 0 and <X_i X_j> = 0, so mean 0 and Var(H) = 12.
#[test]
fn weighted_observable_large_x_group_takes_the_moments_pass() {
    use prism_q::{BackendKind, PauliObservable, PauliTerm};

    let n = 12;
    let c = Circuit::new(n, 0);
    let observable =
        PauliObservable::from_terms((0..n).map(|q| (1.0, vec![PauliTerm::x(q)]))).unwrap();
    assert_eq!(observable.num_groups(), 1);
    let result = sim::simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(common::SEED)
        .observable_expectation(&observable)
        .unwrap();
    assert!(result.mean.abs() < 1e-12);
    assert!((result.variance.unwrap() - 12.0).abs() < 1e-12);
}

// The Z-only analogue on |+>^12: one group of Z_i terms past the pair
// budget, served by the unrotated moments pass. Each <Z_i> = 0 and
// <Z_i Z_j> = 0 on the product state, so mean 0 and Var(H) = 12.
#[test]
fn weighted_observable_large_z_group_takes_the_moments_pass() {
    use prism_q::{BackendKind, PauliObservable, PauliTerm};

    let n = 12;
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    let observable =
        PauliObservable::from_terms((0..n).map(|q| (1.0, vec![PauliTerm::z(q)]))).unwrap();
    assert_eq!(observable.num_groups(), 1);
    let result = sim::simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(common::SEED)
        .observable_expectation(&observable)
        .unwrap();
    assert!(result.mean.abs() < 1e-12);
    assert!((result.variance.unwrap() - 12.0).abs() < 1e-12);
}

// H(0), CX(0,1), S(0) gives (|00> + i|11>)/sqrt(2), a +1 eigenstate of
// Y0 X1: Y0 X1 (|00> + i|11>) = Y0(|01> + i|10>) = i|11> + i(-i)|00> =
// |00> + i|11>. With six spectator X factors on |0>, H = Y0 + X1 + X2..X7
// is one group of 28 pairs, past the pair budget, so the Sdg-H rotated
// moments pass serves it: every single-factor mean is 0 and every cross
// term except <Y0 X1> = 1 vanishes, so mean 0 and Var(H) = 8 + 2 = 10.
// Rotating q0 with the wrong sign (measuring -Y0) would report 6.
#[test]
fn weighted_observable_y_bearing_large_group_variance_pins_the_rotation_sign() {
    use prism_q::{BackendKind, PauliObservable, PauliTerm};

    let n = 8;
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::S, &[0]);
    let mut terms = vec![(1.0, vec![PauliTerm::y(0)])];
    for q in 1..n {
        terms.push((1.0, vec![PauliTerm::x(q)]));
    }
    let observable = PauliObservable::from_terms(terms).unwrap();
    assert_eq!(observable.num_groups(), 1);
    let result = sim::simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(common::SEED)
        .observable_expectation(&observable)
        .unwrap();
    assert!(result.mean.abs() < 1e-12);
    assert!((result.variance.unwrap() - 10.0).abs() < 1e-12);
}

// ---- Pauli rotations ----

// exp(-i t/2 X0 Y1 Z2)|000> = cos(t/2)|000> + sin(t/2)|011>: the string maps
// |000> to X|0> (x) Y|0> (x) Z|0> = i|110> read q0-first, i.e. i times index 3,
// and -i sin(t/2) * i = sin(t/2).
#[test]
fn pauli_rot_xyz_on_zero() {
    use prism_q::PauliTerm;
    let theta = 0.7f64;
    let mut c = Circuit::new(3, 0);
    c.add_pauli_rotation(theta, &[PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)]);
    let sv = run_and_state(&c);
    let (s, cos) = (theta / 2.0).sin_cos();
    for (i, amp) in sv.iter().enumerate() {
        let expected = match i {
            0 => Complex64::new(cos, 0.0),
            3 => Complex64::new(s, 0.0),
            _ => Complex64::new(0.0, 0.0),
        };
        assert_amplitude(*amp, expected, &format!("index {i}"));
    }
}

// exp(-i t/2 X0 X1)|00> = cos(t/2)|00> - i sin(t/2)|11>.
#[test]
fn pauli_rot_xx_on_zero() {
    use prism_q::PauliTerm;
    let theta = 1.1f64;
    let mut c = Circuit::new(2, 0);
    c.add_pauli_rotation(theta, &[PauliTerm::x(0), PauliTerm::x(1)]);
    let sv = run_and_state(&c);
    let (s, cos) = (theta / 2.0).sin_cos();
    assert_amplitude(sv[0], Complex64::new(cos, 0.0), "|00>");
    assert_amplitude(sv[1], Complex64::new(0.0, 0.0), "|01>");
    assert_amplitude(sv[2], Complex64::new(0.0, 0.0), "|10>");
    assert_amplitude(sv[3], Complex64::new(0.0, -s), "|11>");
}

// A Z-only weight-3 string is diagonal: on H^3|000>, every basis state picks
// up e^{-i t/2} at even parity of its three bits and e^{+i t/2} at odd.
#[test]
fn pauli_rot_zzz_applies_parity_phases() {
    use prism_q::PauliTerm;
    let theta = 0.9f64;
    let mut c = Circuit::new(3, 0);
    for q in 0..3 {
        c.add_gate(Gate::H, &[q]);
    }
    c.add_pauli_rotation(theta, &[PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(2)]);
    let sv = run_and_state(&c);
    let amp = 1.0 / (8.0f64).sqrt();
    for (i, actual) in sv.iter().enumerate() {
        let sign = if (i as u32).count_ones().is_multiple_of(2) {
            -1.0
        } else {
            1.0
        };
        let expected = Complex64::from_polar(amp, sign * theta / 2.0);
        assert_amplitude(*actual, expected, &format!("index {i}"));
    }
}

// The recognizing constructor lowers weight-1 strings to the named rotation
// and two-qubit ZZ to Rzz, with factors sorted by qubit.
#[test]
fn pauli_rot_constructor_lowers_recognized_forms() {
    use prism_q::PauliTerm;
    let mut c = Circuit::new(3, 0);
    c.add_pauli_rotation(0.3, &[PauliTerm::x(1)]);
    c.add_pauli_rotation(0.4, &[PauliTerm::y(0)]);
    c.add_pauli_rotation(0.5, &[PauliTerm::z(2)]);
    c.add_pauli_rotation(0.6, &[PauliTerm::z(2), PauliTerm::z(0)]);
    let gates: Vec<_> = c
        .instructions
        .iter()
        .map(|inst| match inst {
            Instruction::Gate { gate, targets } => (gate.clone(), targets.to_vec()),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(gates[0], (Gate::Rx(0.3), vec![1]));
    assert_eq!(gates[1], (Gate::Ry(0.4), vec![0]));
    assert_eq!(gates[2], (Gate::Rz(0.5), vec![2]));
    assert_eq!(gates[3], (Gate::Rzz(0.6), vec![0, 2]));
}

// exp(-i t/2 P) followed by its inverse restores the prepared state exactly
// up to floating-point roundoff.
#[test]
fn pauli_rot_inverse_round_trips() {
    use prism_q::PauliTerm;
    let factors = [PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(3)];
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::Ry(0.4 + 0.2 * q as f64), &[q]);
    }
    c.add_pauli_rotation(0.8, &factors);
    let rot = match c.instructions.last().unwrap() {
        Instruction::Gate { gate, targets } => (gate.inverse(), targets.clone()),
        _ => unreachable!(),
    };
    c.instructions.push(Instruction::Gate {
        gate: rot.0,
        targets: rot.1,
    });

    let mut reference = Circuit::new(4, 0);
    for q in 0..4 {
        reference.add_gate(Gate::Ry(0.4 + 0.2 * q as f64), &[q]);
    }
    let sv = run_and_state(&c);
    let expected = run_and_state(&reference);
    for (i, (a, e)) in sv.iter().zip(&expected).enumerate() {
        assert_amplitude(*a, *e, &format!("index {i}"));
    }
}

// The native kernel against the textbook CNOT-ladder lowering of the same
// string on the same backend, from a non-symmetric product state.
#[test]
fn pauli_rot_matches_ladder_lowering() {
    use prism_q::PauliTerm;
    let factors = [
        PauliTerm::x(0),
        PauliTerm::y(1),
        PauliTerm::z(2),
        PauliTerm::x(3),
    ];
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::Ry(0.3 + 0.15 * q as f64), &[q]);
        c.add_gate(Gate::Rz(0.1 + 0.07 * q as f64), &[q]);
    }
    c.add_pauli_rotation(0.65, &factors);

    let expanded = prism_q::circuit::expand_pauli_rotations(&c);
    assert!(
        matches!(expanded, std::borrow::Cow::Owned(_)),
        "expansion must rewrite the native gate"
    );
    let sv = run_and_state(&c);
    let lowered = run_and_state(&expanded);
    for (i, (a, e)) in sv.iter().zip(&lowered).enumerate() {
        assert_amplitude(*a, *e, &format!("index {i}"));
    }
}
