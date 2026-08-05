//! Golden import tests for OpenQASM 3 exports from common quantum SDKs. Each
//! test parses a representative exported QASM, verifies parse counts, and
//! checks statevector probabilities against an analytic reference.

mod common;

use common::assert_probs_close;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::openqasm;
use prism_q::sim;

fn run_probs(qasm: &str) -> Vec<f64> {
    let circuit = openqasm::parse(qasm).expect("parse");
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).expect("run");
    result.probabilities.expect("probabilities").to_vec()
}

#[test]
fn qiskit_style_qft_3q_with_for_loop() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[3] c;
        h q[0];
        for int i in [1:2] {
            cp(pi / (2 * i)) q[0], q[i];
        }
        h q[1];
        cp(pi / 2) q[1], q[2];
        h q[2];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 3);
    assert_eq!(circuit.num_classical_bits, 3);

    let probs = run_probs(qasm);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!((probs[0] - 0.125).abs() < 1e-10);
}

// Mirrors the shape Qiskit's OpenQASM 3 exporter produces for compiled circuits.
#[test]
fn qiskit_style_def_with_u_gate() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        def my_rx(float t, qubit a) {
            U(t, -pi / 2, pi / 2) a;
        }
        my_rx(pi, q[0]);
        cx q[0], q[1];
    "#;
    let probs = run_probs(qasm);
    assert_probs_close(&probs, &[0.0, 0.0, 0.0, 1.0], 1e-10, "qiskit_def_u");
}

// The shape Qiskit emits when lowering classical feedback after a measurement.
#[test]
fn qiskit_style_conditional_x_after_measure() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        x q[0];
        c[0] = measure q[0];
        if (c[0] == 1) x q[0];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).expect("run");
    let probs = result.probabilities.expect("probs");
    assert!(
        probs.get(0) > 0.999,
        "expected |0> after teleport-style reset"
    );
}

#[test]
fn cirq_style_unrolled_circuit() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        ry(0.7853981633974483) q[0];
        cx q[0], q[1];
        rz(1.5707963267948966) q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.gate_count(), 3);

    let probs = run_probs(qasm);
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
    assert!((probs[0] + probs[3] - 1.0).abs() < 1e-10);
}

// Cirq exports emit these explicit names rather than decomposing them.
#[test]
fn cirq_style_controlled_rotations() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        crx(pi / 3) q[0], q[1];
        cry(pi / 4) q[0], q[1];
        crz(pi / 5) q[0], q[1];
        swap q[0], q[1];
    "#;
    let probs = run_probs(qasm);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

// `gpi`, `gpi2`, and `ms` (Mølmer-Sørensen) are IonQ's native instruction
// set; their cloud transpiler emits these directly.
#[test]
fn ionq_style_native_gates() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        gpi(0.0) q[0];
        gpi2(0.25) q[1];
        ms(0.0, 0.0, 0.25) q[0], q[1];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 2);

    let probs = run_probs(qasm);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

// IonQ's compiler emits hex-prefix integer literals for register
// comparisons in feedforward circuits.
#[test]
fn ionq_style_conditional_with_hex_literal() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        if (c == 0x3) x q[0];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.num_classical_bits, 2);
}

// `syc` and `sqrt_iswap` are Google's hardware-native two-qubit gates,
// exposed by Cirq's OQ3 export when targeting Sycamore-class processors.
#[test]
fn google_style_sycamore_gates() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        syc q[0], q[1];
        sqrt_iswap q[0], q[1];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 2);

    let probs = run_probs(qasm);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

// `cphase` is the form Google's exporter prefers over the Qiskit `cp` alias.
#[test]
fn google_style_qft_with_cphase_alias() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        h q[0];
        cphase(pi / 2) q[0], q[1];
        cphase(pi / 4) q[0], q[2];
        h q[1];
        cphase(pi / 2) q[1], q[2];
        h q[2];
        swap q[0], q[2];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 3);

    let probs = run_probs(qasm);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!((probs[0] - 0.125).abs() < 1e-10);
}

// Mixes for-loop unrolling, a parametric def, and a binary integer literal:
// the structure Qiskit produces when exporting a compiled QAOA layer.
#[test]
fn qiskit_style_qaoa_layer_with_for_and_def() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[4] q;
        def zz_layer(float gamma, qubit a, qubit b) {
            cx a, b;
            rz(gamma) b;
            cx a, b;
        }
        for int i in [0:3] {
            h q[i];
        }
        for int i in [0:2] {
            zz_layer(0b1 * 0.4, q[i], q[i + 1]);
        }
        for int i in [0:3] {
            rx(0.3) q[i];
        }
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 4);

    let probs = run_probs(qasm);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

// Qiskit's older 2.0 exporter still produces qreg/creg forms in the wild;
// OQ3 backward-compat keeps them parsing.
#[test]
fn qiskit_legacy_qreg_creg_style() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        u3(pi / 2, 0, pi) q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        if (c == 3) x q[0];
    "#;
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.num_classical_bits, 2);
}
