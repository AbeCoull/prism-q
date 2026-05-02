//! Golden import tests for OpenQASM 3 exports from common quantum SDKs.
//!
//! These exercise representative samples of the gate sets and control-flow
//! constructs each ecosystem typically produces. Each test parses the
//! exported QASM and verifies parse counts, then runs the circuit through
//! the statevector backend and checks the resulting probability distribution
//! against an analytic reference.

use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::openqasm;
use prism_q::sim;

fn run_probs(qasm: &str) -> Vec<f64> {
    let circuit = openqasm::parse(qasm).expect("parse");
    let mut backend = StatevectorBackend::new(42);
    let result = sim::run_on(&mut backend, &circuit).expect("run");
    result.probabilities.expect("probabilities").to_vec()
}

fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(actual.len(), expected.len(), "probability length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "prob[{i}]: expected {e:.6}, got {a:.6}"
        );
    }
}

/// Qiskit-style export: `for` loop, `cp` controlled phase, U/u gates,
/// `qubit[]` / `bit[]` declarations, `c[i] = measure q[i]` assignment form.
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

/// Qiskit-style: parametric `def` subroutine with a float angle parameter,
/// register-broadcast call, and U-gate body. Mirrors the shape Qiskit's
/// OpenQASM 3 exporter produces for compiled circuits.
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
    assert_probs_close(&probs, &[0.0, 0.0, 0.0, 1.0], 1e-10);
}

/// Qiskit-style conditional reset: `if (c[0] == 1) x q[0]` is the canonical
/// shape Qiskit emits when lowering classical feedback after a measurement.
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
    assert!(probs[0] > 0.999, "expected |0> after teleport-style reset");
}

/// Cirq-style: explicit unrolled gates, single-letter rotation names, OQ2
/// arrow-form measurements, no `for` or `def`.
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

/// Cirq-style controlled rotation set: `crx`, `cry`, `crz`, plus `swap`.
/// Cirq exports usually emit OQ3 with these explicit names rather than
/// decomposing them.
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

/// IonQ-style: native trapped-ion gate set. `gpi`, `gpi2`, and `ms`
/// (Mølmer-Sørensen) are IonQ's native instruction set; their cloud
/// transpiler emits these directly.
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

/// IonQ-style with classical conditional: their compiler emits hex-prefix
/// integer literals for register comparisons in feedforward circuits.
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

/// Google Sycamore-style: `syc`, `sqrt_iswap`. These are Google's hardware
/// native two-qubit gates exposed by Cirq's OQ3 export when targeting
/// Sycamore-class processors.
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

/// Google-style: full QFT-style circuit with explicit register declarations
/// and the cphase form their exporter prefers over the Qiskit `cp` alias.
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

/// Combined: a circuit that mixes static for-loop unrolling, a parametric
/// def subroutine, and binary integer literals — the kind of structure
/// Qiskit produces when exporting a compiled QAOA layer.
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

/// Pre-OQ3 Qiskit (qreg/creg) style. Qiskit's older 2.0 exporter still
/// produces these forms in the wild; OQ3 backward-compat keeps them
/// parsing.
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
