//! Smoke tests: OpenQASM 3.0 parsing → statevector simulation round-trips.

use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::openqasm;
use prism_q::sim;

fn assert_probs(probs: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(probs.len(), expected.len());
    for (i, (a, e)) in probs.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "prob[{i}]: expected {e:.6}, got {a:.6}"
        );
    }
}

#[test]
fn bell_state() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.num_classical_bits, 2);
    assert_eq!(circuit.gate_count(), 2);

    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probs = result.probabilities.unwrap().to_vec();

    assert_probs(&probs, &[0.5, 0.0, 0.0, 0.5], 1e-10);
}

#[test]
fn ghz_3_qubit() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        h q[0];
        cx q[0], q[1];
        cx q[0], q[2];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probs = result.probabilities.unwrap().to_vec();

    assert_probs(&probs, &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], 1e-10);
}

#[test]
fn single_x_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nx q[0];";

    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probs = result.probabilities.unwrap().to_vec();

    assert_probs(&probs, &[0.0, 1.0], 1e-10);
}

#[test]
fn parametric_rx_pi() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(pi) q[0];";

    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probs = result.probabilities.unwrap().to_vec();

    assert_probs(&probs, &[0.0, 1.0], 1e-10);
}

#[test]
fn measure_deterministic_oq3() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        x q[0];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();

    assert!(result.classical_bits[0]);
    assert!(!result.classical_bits[1]);
}

#[test]
fn multiple_registers() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] a;
        qubit[1] b;
        bit[2] ca;
        bit[1] cb;
        h a[0];
        cx a[0], a[1];
        x b[0];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    assert_eq!(circuit.num_qubits, 3);
    assert_eq!(circuit.num_classical_bits, 3);

    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probs = result.probabilities.unwrap().to_vec();

    assert_probs(&probs, &[0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5], 1e-10);
}

#[test]
fn run_qasm_convenience() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
    "#;

    let result = prism_q::run_qasm(qasm, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    assert_probs(&probs, &[0.5, 0.5], 1e-10);
}

#[test]
fn parse_error_undefined_register() {
    let qasm = "OPENQASM 3.0;\nh q[0];";
    let err = openqasm::parse(qasm).unwrap_err();
    assert!(
        format!("{err}").contains("undefined register"),
        "expected UndefinedRegister error, got: {err}"
    );
}

#[test]
fn parse_error_unsupported_construct() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nwhile (true) { x q[0]; }";
    let err = openqasm::parse(qasm).unwrap_err();
    assert!(
        format!("{err}").contains("unsupported"),
        "expected UnsupportedConstruct error, got: {err}"
    );
}

#[test]
fn all_single_qubit_gates_parse() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        id q[0];
        x q[0];
        y q[0];
        z q[0];
        h q[0];
        s q[0];
        sdg q[0];
        t q[0];
        tdg q[0];
        rx(0.5) q[0];
        ry(pi/2) q[0];
        rz(2*pi) q[0];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    assert_eq!(circuit.gate_count(), 12);
}

#[test]
fn all_two_qubit_gates_parse() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        cx q[0], q[1];
        cz q[0], q[1];
        swap q[0], q[1];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    assert_eq!(circuit.gate_count(), 3);
}

#[test]
fn ecosystem_gate_bundle_parses_and_runs() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[5] q;
        h q[0];
        r(pi/3, pi/7) q[0];
        phase(pi/9) q[1];
        gpi(0.25) q[2];
        gpi2(0.5) q[3];
        xx_plus_yy(pi/5, pi/11) q[0], q[1];
        xx_minus_yy(pi/7, pi/13) q[1], q[2];
        syc q[2], q[3];
        sqrt_iswap q[3], q[4];
        sqrt_iswap_inv q[3], q[4];
        ms(0.125, 0.25, 0.125) q[0], q[4];
        cs q[0], q[1];
        csdg q[1], q[2];
        cu(pi/3, pi/5, pi/7, pi/11) q[2], q[3];
        c3x q[0], q[1], q[2], q[3];
        c4x q[0], q[1], q[2], q[3], q[4];
        rccx q[0], q[1], q[2];
        rcccx q[0], q[1], q[2], q[3];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probabilities = result.probabilities.unwrap();
    let total: f64 = probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

// -- Gate modifier tests --

#[test]
fn modifier_inv_h_is_h() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\ninv @ h q[0];";
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    assert_probs(&result.probabilities.unwrap().to_vec(), &[0.5, 0.5], 1e-10);
}

#[test]
fn modifier_inv_t_is_tdg() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
        inv @ t q[0];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, &circuit).unwrap();

    let qasm2 = r#"
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
        tdg q[0];
    "#;
    let circuit2 = openqasm::parse(qasm2).unwrap();
    let mut backend2 = StatevectorBackend::new(42);
    sim::run_on(&mut backend2, &circuit2).unwrap();

    let sv1 = backend.state_vector();
    let sv2 = backend2.state_vector();
    for (a, b) in sv1.iter().zip(sv2) {
        assert!((a - b).norm() < 1e-10, "inv @ t != tdg: {a} vs {b}");
    }
}

#[test]
fn modifier_ctrl_x_is_cx() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        h q[0];
        ctrl @ x q[0], q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    assert_probs(
        &result.probabilities.unwrap().to_vec(),
        &[0.5, 0.0, 0.0, 0.5],
        1e-10,
    );
}

#[test]
fn modifier_ctrl_h_controlled_hadamard() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        x q[0];
        ctrl @ h q[0], q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    // x q[0] → |01⟩ (q0=1, LSB). ctrl @ h with ctrl=q0 active → H on q1.
    // |01⟩ → (|01⟩ + |11⟩)/√2 → indices 1, 3
    assert_probs(
        &result.probabilities.unwrap().to_vec(),
        &[0.0, 0.5, 0.0, 0.5],
        1e-10,
    );
}

#[test]
fn modifier_ctrl_rz_controlled_phase() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        x q[0];
        x q[1];
        ctrl @ rz(pi) q[0], q[1];
    "#;
    // |11⟩ → CRz(π) on q1: Rz(π)|1⟩ = e^{iπ/2}|1⟩ = i|1⟩
    // State should still be |11⟩ with amplitude i (phase only)
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, &circuit).unwrap();
    let probs = backend.probabilities().unwrap();
    assert_probs(&probs, &[0.0, 0.0, 0.0, 1.0], 1e-10);
}

#[test]
fn modifier_inv_ctrl_rx() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        x q[0];
        ctrl @ rx(pi/4) q[0], q[1];
        inv @ ctrl @ rx(pi/4) q[0], q[1];
    "#;
    // Apply Cu(Rx(π/4)) then inv = Cu(Rx(-π/4)); should cancel to |01⟩
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, &circuit).unwrap();
    let probs = backend.probabilities().unwrap();
    assert_probs(&probs, &[0.0, 1.0, 0.0, 0.0], 1e-10);
}

#[test]
fn modifier_pow_2_t_is_s() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
        pow(2) @ t q[0];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, &circuit).unwrap();

    let qasm2 = r#"
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
        s q[0];
    "#;
    let circuit2 = openqasm::parse(qasm2).unwrap();
    let mut backend2 = StatevectorBackend::new(42);
    sim::run_on(&mut backend2, &circuit2).unwrap();

    let sv1 = backend.state_vector();
    let sv2 = backend2.state_vector();
    for (a, b) in sv1.iter().zip(sv2) {
        assert!((a - b).norm() < 1e-10, "pow(2) @ t != s: {a} vs {b}");
    }
}

#[test]
fn modifier_pow_3_x_is_x() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\npow(3) @ x q[0];";
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    // X^3 = X (since X^2 = I)
    assert_probs(&result.probabilities.unwrap().to_vec(), &[0.0, 1.0], 1e-10);
}

#[test]
fn modifier_sparse_cu_matches_statevector() {
    use prism_q::backend::sparse::SparseBackend;
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        x q[0];
        ctrl @ h q[0], q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();

    let mut sv = StatevectorBackend::new(42);
    sim::run_on(&mut sv, &circuit).unwrap();
    let sv_probs = sv.probabilities().unwrap();

    let mut sp = SparseBackend::new(42);
    sim::run_on(&mut sp, &circuit).unwrap();
    let sp_probs = sp.probabilities().unwrap();

    assert_probs(&sp_probs, &sv_probs, 1e-10);
}

#[test]
fn modifier_mps_cu_matches_statevector() {
    use prism_q::backend::mps::MpsBackend;
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        x q[0];
        ctrl @ h q[0], q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();

    let mut sv = StatevectorBackend::new(42);
    sim::run_on(&mut sv, &circuit).unwrap();
    let sv_probs = sv.probabilities().unwrap();

    let mut mps = MpsBackend::new(42, 64);
    sim::run_on(&mut mps, &circuit).unwrap();
    let mps_probs = mps.probabilities().unwrap();

    assert_probs(&mps_probs, &sv_probs, 1e-10);
}

#[test]
fn modifier_ctrl_ctrl_x_toffoli() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        x q[0];
        x q[1];
        ctrl @ ctrl @ x q[0], q[1], q[2];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    // Both controls active → X on target: |011⟩ → |111⟩
    assert_probs(
        &result.probabilities.unwrap().to_vec(),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        1e-10,
    );
}

#[test]
fn modifier_ctrl_ctrl_x_no_action() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        x q[0];
        ctrl @ ctrl @ x q[0], q[1], q[2];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    // Only one control active → no X on target: stays |001⟩
    assert_probs(
        &result.probabilities.unwrap().to_vec(),
        &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1e-10,
    );
}

#[test]
fn modifier_ctrl_swap_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ swap q[0], q[1], q[2];";
    let err = openqasm::parse(qasm).unwrap_err();
    assert!(
        format!("{err}").contains("unsupported"),
        "expected unsupported, got: {err}"
    );
}

#[test]
fn oq2_backward_compat() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.gate_count(), 2);
    assert_eq!(circuit.instructions.len(), 4);
}

#[test]
fn conditional_gate_executed() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[1] c;
        x q[0];
        c[0] = measure q[0];
        if (c[0]) x q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let result = sim::run(&circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    // q[0] measured |1⟩, so c[0]=1, conditional fires, q[1] → |1⟩
    // After measurement q[0] collapsed to |1⟩, q[1] flipped to |1⟩ → state |11⟩ = index 3
    assert!((probs[3] - 1.0).abs() < 1e-10);
}

#[test]
fn conditional_gate_not_executed() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[1] c;
        c[0] = measure q[0];
        if (c[0]) x q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let result = sim::run(&circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    // q[0] starts as |0⟩, measured → c[0]=0, conditional doesn't fire
    // State stays |00⟩ = index 0
    assert!((probs[0] - 1.0).abs() < 1e-10);
}

#[test]
fn conditional_oq2_register_equals() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];
        x q[0];
        measure q[0] -> c[0];
        if(c==1) x q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let result = sim::run(&circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    // c[0]=1 (from X then measure), c[1]=0 → c = 0b01 = 1
    // Condition c==1 is true → X applied to q[1] → state |11⟩
    assert!((probs[3] - 1.0).abs() < 1e-10);
}

#[test]
fn conditional_oq2_register_no_match() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];
        x q[0];
        measure q[0] -> c[0];
        if(c==2) x q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let result = sim::run(&circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    // c = 0b01 = 1, condition c==2 is false → q[1] stays |0⟩
    // q[0] collapsed to |1⟩ → state |01⟩ = index 1
    assert!((probs[1] - 1.0).abs() < 1e-10);
}

// ---- Multi-line gate definition ----

#[test]
fn multi_line_gate_body() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate bell a, b {
            h a;
            cx a, b;
        }
        bell q[0], q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    assert_probs(&probs, &[0.5, 0.0, 0.0, 0.5], 1e-10);
}

#[test]
fn multi_line_parametric_gate() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate myrot(theta) a {
            rx(theta) a;
        }
        myrot(pi) q[0];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    assert_probs(&probs, &[0.0, 1.0], 1e-10);
}

// ---- Broadcast measure assign ----

#[test]
fn broadcast_measure_all() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        x q[0];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let result = sim::run(&circuit, 42).unwrap();
    assert!(result.classical_bits[0]);
    assert!(!result.classical_bits[1]);
}

// ---- Out-of-bounds qubit index ----

#[test]
fn oob_qubit_index_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[5];";
    let err = openqasm::parse(qasm).unwrap_err();
    assert!(
        format!("{err}").contains("invalid qubit index"),
        "expected InvalidQubit error, got: {err}"
    );
}

#[test]
fn oob_classical_bit_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nc[3] = measure q[0];";
    let err = openqasm::parse(qasm).unwrap_err();
    assert!(
        format!("{err}").contains("invalid classical bit"),
        "expected InvalidClassicalBit error, got: {err}"
    );
}

// ---- run_shots through QASM ----

#[test]
fn run_shots_via_qasm() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        x q[0];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let result = prism_q::run_shots(&circuit, 100, 42).unwrap();
    let counts = result.counts();
    assert_eq!(counts.len(), 1);
    assert_eq!(*counts.get(&vec![1u64]).unwrap_or(&0), 100);
}

#[test]
fn run_shots_superposition_via_qasm() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        h q[0];
        c[0] = measure q[0];
    "#;
    let circuit = openqasm::parse(qasm).unwrap();
    let result = prism_q::run_shots(&circuit, 1000, 42).unwrap();
    let counts = result.counts();
    let zeros = *counts.get(&vec![0u64]).unwrap_or(&0);
    let ones = *counts.get(&vec![1u64]).unwrap_or(&0);
    assert!(zeros > 100, "expected >100 zeros, got {zeros}");
    assert!(ones > 100, "expected >100 ones, got {ones}");
    assert_eq!(zeros + ones, 1000);
}
