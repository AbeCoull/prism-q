//! Golden import tests for OpenQASM 3 exports from common quantum SDKs. Each
//! test parses a representative exported QASM, verifies parse counts, and
//! checks statevector probabilities against an analytic reference. The export
//! half round-trips the same programs and the generated corpus back out.

mod common;
mod conformance_corpus;

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

// ---- Export round-trip ----

use common::{SV_EPS, sv_reference_probs};
use conformance_corpus::generated_cases;
use num_complex::Complex64;
use prism_q::PrismError;
use prism_q::circuit::qasm_export::to_qasm3;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::gates::Gate;

/// Matrix payloads are rebuilt from recovered Euler angles, so the round trip
/// reproduces them to round-off rather than bit for bit. Inline angles and
/// every other field match exactly.
const PAYLOAD_EPS: f64 = 1e-12;

fn round_trip(circuit: &Circuit) -> Circuit {
    let qasm = to_qasm3(circuit).expect("export");
    openqasm::parse(&qasm).unwrap_or_else(|err| panic!("reparse failed: {err}\n{qasm}"))
}

fn assert_streams_match(original: &Circuit, round: &Circuit, label: &str) {
    assert_eq!(
        original.num_qubits, round.num_qubits,
        "{label}: qubit count"
    );
    assert_eq!(
        original.num_classical_bits, round.num_classical_bits,
        "{label}: classical bit count"
    );
    assert_eq!(
        original.instructions.len(),
        round.instructions.len(),
        "{label}: instruction count"
    );
    for (i, (a, b)) in original
        .instructions
        .iter()
        .zip(&round.instructions)
        .enumerate()
    {
        assert!(
            instructions_match(a, b),
            "{label}: instruction {i} differs\n  before: {a:?}\n  after:  {b:?}"
        );
    }
}

fn instructions_match(a: &Instruction, b: &Instruction) -> bool {
    match (a, b) {
        (
            Instruction::Gate {
                gate: ga,
                targets: ta,
            },
            Instruction::Gate {
                gate: gb,
                targets: tb,
            },
        ) => ta == tb && gates_match(ga, gb),
        (
            Instruction::Measure {
                qubit: qa,
                classical_bit: ca,
            },
            Instruction::Measure {
                qubit: qb,
                classical_bit: cb,
            },
        ) => qa == qb && ca == cb,
        (Instruction::Reset { qubit: qa }, Instruction::Reset { qubit: qb }) => qa == qb,
        (Instruction::Region(_), Instruction::Region(_)) => format!("{a:?}") == format!("{b:?}"),
        (Instruction::Barrier { qubits: qa }, Instruction::Barrier { qubits: qb }) => qa == qb,
        (
            Instruction::Conditional {
                condition: ca,
                gate: ga,
                targets: ta,
            },
            Instruction::Conditional {
                condition: cb,
                gate: gb,
                targets: tb,
            },
        ) => ta == tb && gates_match(ga, gb) && format!("{ca:?}") == format!("{cb:?}"),
        _ => false,
    }
}

fn gates_match(a: &Gate, b: &Gate) -> bool {
    if std::mem::discriminant(a) != std::mem::discriminant(b) {
        return false;
    }
    match (a, b) {
        (Gate::Fused(x), Gate::Fused(y)) | (Gate::Cu(x), Gate::Cu(y)) => {
            close(x.iter().flatten(), y.iter().flatten())
        }
        (Gate::Fused2q(x), Gate::Fused2q(y)) => close(x.iter().flatten(), y.iter().flatten()),
        (Gate::Mcu(x), Gate::Mcu(y)) => {
            x.num_controls == y.num_controls
                && close(x.mat.iter().flatten(), y.mat.iter().flatten())
        }
        _ => a == b,
    }
}

fn close<'a>(
    a: impl Iterator<Item = &'a num_complex::Complex64>,
    b: impl Iterator<Item = &'a num_complex::Complex64>,
) -> bool {
    a.zip(b).all(|(x, y)| (x - y).norm() < PAYLOAD_EPS)
}

// The corpus `conformance_matrix.rs` runs across backends, reused here as the
// round-trip measure: it reaches measurement, reset, and both condition shapes,
// which the SDK programs above do not all carry.
#[test]
fn export_round_trips_the_generated_corpus() {
    for case in generated_cases() {
        let round = round_trip(&case.circuit);
        assert_streams_match(&case.circuit, &round, &case.name());
        assert_probs_close(
            &sv_reference_probs(&round),
            &sv_reference_probs(&case.circuit),
            SV_EPS,
            &case.name(),
        );
    }
}

#[test]
fn export_round_trips_the_sdk_programs() {
    let programs = [
        (
            "ionq_native",
            r#"
            OPENQASM 3.0;
            qubit[2] q;
            gpi(0.0) q[0];
            gpi2(0.25) q[1];
            ms(0.1, 0.2, 0.25) q[0], q[1];
            "#,
        ),
        (
            "google_sycamore",
            r#"
            OPENQASM 3.0;
            qubit[2] q;
            syc q[0], q[1];
            sqrt_iswap q[0], q[1];
            sqrt_iswap_inv q[0], q[1];
            "#,
        ),
        (
            "controlled_rotations",
            r#"
            OPENQASM 3.0;
            qubit[3] q;
            crx(pi / 3) q[0], q[1];
            cry(pi / 4) q[0], q[1];
            crz(pi / 5) q[0], q[1];
            cp(pi / 2) q[0], q[2];
            ch q[1], q[2];
            ccx q[0], q[1], q[2];
            "#,
        ),
        (
            "xy_interactions",
            r#"
            OPENQASM 3.0;
            qubit[2] q;
            xx_plus_yy(0.6, 0.3) q[0], q[1];
            xx_minus_yy(0.9, -0.4) q[0], q[1];
            rzz(0.31) q[0], q[1];
            "#,
        ),
        (
            "qiskit_legacy_feedforward",
            r#"
            OPENQASM 2.0;
            qreg q[2];
            creg c[2];
            u3(pi / 2, 0, pi) q[0];
            barrier q[0], q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
            if (c == 3) x q[0];
            if (c != 1) z q[1];
            "#,
        ),
        (
            "mid_circuit_reset",
            r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            c[0] = measure q[0];
            reset q[0];
            if (c[0]) x q[1];
            if (!c[1]) y q[1];
            "#,
        ),
    ];

    for (label, qasm) in programs {
        let circuit = openqasm::parse(qasm).expect("parse");
        assert_streams_match(&circuit, &round_trip(&circuit), label);
    }
}

// q[0] is the LSB: an export that reverses qubit order passes a shape check and
// fails this one.
#[test]
fn export_preserves_qubit_order() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::X, &[0]);
    let qasm = to_qasm3(&circuit).expect("export");
    assert!(qasm.contains("x q[0];"), "{qasm}");

    let probs = sv_reference_probs(&openqasm::parse(&qasm).expect("reparse"));
    assert!(
        (probs[1] - 1.0).abs() < SV_EPS,
        "expected |001>, got {probs:?}"
    );
}

#[test]
fn export_expands_a_qft_block() {
    let circuit = prism_q::circuits::qft_circuit(4);
    let round = round_trip(&circuit);
    assert!(round.gate_count() > 1);
    assert_probs_close(
        &sv_reference_probs(&round),
        &sv_reference_probs(&circuit),
        SV_EPS,
        "qft_4",
    );
}

#[test]
fn export_rejects_a_fused_circuit() {
    let circuit = prism_q::circuits::hardware_efficient_ansatz(16, 2, 42);
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    assert!(to_qasm3(&circuit).is_ok());
    assert!(matches!(
        to_qasm3(&fused),
        Err(PrismError::ExportUnsupported { .. })
    ));
}

// The parameter surface hands out an unfused circuit per binding, so a swept
// point exports like any other circuit.
#[test]
fn export_round_trips_a_bound_parameter_point() {
    let mut template = Circuit::new(3, 0);
    template.add_gate(Gate::Ry(0.0), &[0]);
    template.add_gate(Gate::Cx, &[0, 1]);
    template.add_gate(Gate::Rzz(0.0), &[1, 2]);
    let params = prism_q::Parameters::all_rotations(&template);
    let mut prepared = prism_q::PreparedCircuit::new(template, params).expect("prepare");

    let bound = prepared.bind(&[0.41, 1.27]).expect("bind").clone();
    assert_streams_match(&bound, &round_trip(&bound), "bound_point");
}

// Every classical-control construct the parser module header lists, with the
// outcome it promises: parse, or reject by name with a line number. The
// rejections are the half worth pinning, because a construct that falls
// through to gate parsing reports a register error naming a brace.
#[test]
fn classical_control_constructs_parse_or_reject_by_name() {
    const PROLOGUE: &str = "OPENQASM 3.0;\nqubit[3] q;\nbit[2] c;\n";

    let accepted = [
        ("guarded gate", "if (c[0]) x q[0];"),
        ("guarded gate, legacy register", "if (c == 1) x q[0];"),
        (
            "guarded region",
            "if (c[0]) { x q[0]; measure q[1] -> c[1]; }",
        ),
        ("negated bit", "if (!c[0]) x q[0];"),
        ("bit literal", "if (c[0] == 1) x q[0];"),
        ("register inequality", "if (c != 0) x q[0];"),
        ("parity", "if (c[0] ^ c[1]) x q[0];"),
        ("parity against zero", "if ((c[0] ^ c[1]) == 0) x q[0];"),
        ("else, one line", "if (c[0]) x q[0]; else z q[0];"),
        ("else, braced", "if (c[0]) { x q[0]; } else { z q[0]; }"),
        (
            "else on its own line",
            "if (c[0]) { x q[0]; }\nelse\nz q[0];",
        ),
        (
            "else if",
            "if (c[0]) { x q[0]; } else if (c[1]) { z q[0]; }",
        ),
        (
            "switch",
            "switch (c) { case 0 { x q[0]; } case 1 { z q[0]; } }",
        ),
        (
            "switch with default",
            "switch (c) { case 0 { x q[0]; } default { h q[0]; } }",
        ),
        ("bounded for", "for int i in [0:2] { x q[i]; }"),
    ];
    for (label, body) in accepted {
        let qasm = format!("{PROLOGUE}{body}");
        assert!(
            openqasm::parse(&qasm).is_ok(),
            "{label} should parse: {qasm}"
        );
    }

    let rejected = [
        ("while", "while (c[0]) { x q[0]; }", "while"),
        ("break", "break;", "break"),
        ("box", "box { x q[0]; }", "box"),
        ("defcal", "defcal x $0 { }", "defcal"),
        ("opaque", "opaque foo q;", "opaque"),
        ("extern", "extern foo(int);", "extern"),
        ("return", "return;", "return"),
        ("stray else", "else { x q[0]; }", "else"),
    ];
    for (label, body, construct) in rejected {
        let qasm = format!("{PROLOGUE}{body}");
        match openqasm::parse(&qasm) {
            Err(PrismError::UnsupportedConstruct {
                construct: got,
                line,
            }) => {
                assert_eq!(got, construct, "{label}");
                assert_eq!(line, 4, "{label} should report its own line");
            }
            other => panic!("{label} should reject by name, got {other:?}"),
        }
    }

    // Rejected for a reason the construct name does not carry, so these report a
    // parse error with the line and the reason instead.
    let explained = [
        (
            "else whose body overwrites its guard",
            "if (c[0]) { measure q[0] -> c[0]; } else { x q[1]; }",
            "overwrite",
        ),
        (
            "switch arm writing the switched register",
            "switch (c) { case 1 { measure q[0] -> c[0]; } case 2 { x q[0]; } }",
            "overwrites",
        ),
        (
            "duplicate case label",
            "switch (c) { case 1 { x q[0]; } case 1 { z q[0]; } }",
            "twice",
        ),
        (
            "switch body that is not an arm",
            "switch (c) { x q[0]; }",
            "`case` or `default`",
        ),
    ];
    for (label, body, needle) in explained {
        let qasm = format!("{PROLOGUE}{body}");
        match openqasm::parse(&qasm) {
            Err(PrismError::Parse { message, line }) => {
                assert!(message.contains(needle), "{label}: {message}");
                assert_eq!(line, 4, "{label} should report its own line");
            }
            other => panic!("{label} should be a parse error, got {other:?}"),
        }
    }
}

const PARAMETRIC: &str = r#"
OPENQASM 3.0;
include "stdgates.inc";
input float[64] theta;
input angle phi;
qubit[3] q;
output bit[3] c;
h q[0];
rx(theta) q[0];
cx q[0], q[1];
rz(phi) q[1];
rzz(theta) q[1], q[2];
"#;

// The deliverable is a round trip rather than a parse: an `input` declaration
// becomes a named slot, a value vector binds through it, and the bound point
// exports and re-imports as the same instruction stream.
#[test]
fn an_input_declaration_round_trips_through_a_bound_point() {
    let (template, params) = openqasm::parse_parametric(PARAMETRIC).expect("parse");
    assert_eq!(template.num_qubits, 3);
    assert_eq!(template.num_classical_bits, 3);
    assert_eq!(params.num_slots(), 2);

    let bound = params.bind(&template, &[0.41, 1.27]).expect("bind");
    assert_eq!(params.values(&bound).expect("values"), vec![0.41, 1.27]);

    let round = round_trip(&bound);
    assert_streams_match(&bound, &round, "input_bound_point");

    // The exported text carries the bound angles, not the declarations, so the
    // reparse needs no parameter surface of its own.
    let qasm = to_qasm3(&bound).expect("export");
    assert!(!qasm.contains("input"), "export emitted an input: {qasm}");
    assert_probs_close(
        &sv_reference_probs(&round),
        &sv_reference_probs(&bound),
        SV_EPS,
        "input_bound_point",
    );
}

#[test]
fn input_slots_are_named_and_ordered_by_declaration() {
    let (template, params) = openqasm::parse_parametric(PARAMETRIC).expect("parse");
    assert_eq!(params.name_of(0), Some("theta"));
    assert_eq!(params.name_of(1), Some("phi"));
    assert_eq!(params.slot_of("phi"), Some(1));
    assert_eq!(params.slot_of("psi"), None);
    assert!(params.unread_slots().is_empty());

    // `theta` drives two gates and `phi` one, which is the weight sharing the
    // parameter surface already models.
    let mut per_slot = [0usize; 2];
    for link in params.links() {
        per_slot[link.slot] += 1;
    }
    assert_eq!(per_slot, [2, 1]);
    params.validate(&template).expect("links validate");
}

#[test]
fn a_bound_program_agrees_with_the_same_angles_written_out() {
    let (template, params) = openqasm::parse_parametric(PARAMETRIC).expect("parse");
    let bound = params.bind(&template, &[0.41, 1.27]).expect("bind");

    let literal = openqasm::parse(
        r#"
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        h q[0];
        rx(0.41) q[0];
        cx q[0], q[1];
        rz(1.27) q[1];
        rzz(0.41) q[1], q[2];
        "#,
    )
    .expect("parse literal");
    assert_probs_close(
        &sv_reference_probs(&bound),
        &sv_reference_probs(&literal),
        SV_EPS,
        "bound_vs_literal",
    );
}

#[test]
fn parse_rejects_a_program_that_leaves_inputs_unbound() {
    let err = openqasm::parse(PARAMETRIC).expect_err("unbound inputs");
    assert!(
        matches!(err, PrismError::InvalidParameter { .. }),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn an_input_broadcast_over_a_register_shares_one_slot() {
    let qasm = "OPENQASM 3.0;\ninput float[64] t;\nqubit[3] q;\nrx(t) q;\n";
    let (template, params) = openqasm::parse_parametric(qasm).expect("parse");
    assert_eq!(params.num_slots(), 1);
    assert_eq!(params.links().len(), 3);
    assert!(params.links().iter().all(|l| l.slot == 0));

    let bound = params.bind(&template, &[0.9]).expect("bind");
    let literal = openqasm::parse(
        "OPENQASM 3.0;\nqubit[3] q;\nrx(0.9) q[0];\nrx(0.9) q[1];\nrx(0.9) q[2];\n",
    )
    .expect("parse literal");
    assert_probs_close(
        &sv_reference_probs(&bound),
        &sv_reference_probs(&literal),
        SV_EPS,
        "broadcast_vs_literal",
    );
}

// Every way an `input` can be written that the binding surface cannot carry.
// One angle is written per link, so an input reaching anything but a whole
// top-level rotation angle has nowhere to land, and a silent wrong angle is the
// failure worth pinning against.
#[test]
fn input_uses_the_binding_surface_cannot_carry_reject_by_name() {
    const PROLOGUE: &str = "OPENQASM 3.0;\ninput float[64] t;\nqubit[3] q;\nbit[3] c;\n";

    let rejected = [
        ("expression over an input", "rx(2 * t) q[0];"),
        ("expression with a function", "rx(sin(t)) q[0];"),
        (
            "two inputs on one gate",
            "input float[64] u;\nr(t, u) q[0];",
        ),
        ("gate carrying no angle", "cu(t, 0, 0, 0) q[0], q[1];"),
        ("modified gate", "ctrl @ rx(t) q[0], q[1];"),
        ("inside a for body", "for int i in [0:1] { rx(t) q[i]; }"),
        ("inside a guarded region", "if (c[0]) { rx(t) q[0]; }"),
        ("inside a guarded statement", "if (c[0]) rx(t) q[0];"),
        (
            "user-defined gate",
            "gate myrot(a) x { rx(2 * a) x; }
myrot(t) q[0];",
        ),
        ("lowered gate", "xx_plus_yy(t, 0.2) q[0], q[1];"),
        (
            "inside a def body",
            "def d(qubit a, float w) { rx(w) a; }
d(q[0], t);",
        ),
    ];
    for (label, body) in rejected {
        let qasm = format!("{PROLOGUE}{body}");
        let err = openqasm::parse_parametric(&qasm)
            .err()
            .unwrap_or_else(|| panic!("{label}: expected a rejection, parsed"));
        assert!(
            matches!(err, PrismError::UnsupportedConstruct { .. }),
            "{label}: expected UnsupportedConstruct, got {err:?}"
        );
    }

    let bad_types = [
        (
            "integer input",
            "OPENQASM 3.0;\ninput int[32] n;\nqubit[1] q;\n",
        ),
        (
            "qubit output",
            "OPENQASM 3.0;\nqubit[1] q;\noutput float[64] x;\n",
        ),
    ];
    for (label, qasm) in bad_types {
        assert!(
            matches!(
                openqasm::parse_parametric(qasm),
                Err(PrismError::UnsupportedConstruct { .. })
            ),
            "{label}: expected UnsupportedConstruct"
        );
    }

    assert!(
        openqasm::parse_parametric("OPENQASM 3.0;\ninput float[64] t;\ninput float[64] t;\n")
            .is_err(),
        "a name declared twice should not silently take the second slot"
    );
}

// A Pauli rotation is one gate carrying one angle, so an `input` binds it the
// way it binds `rx`: the spelling exists and the binding surface reaches it.
#[test]
fn an_input_binds_a_pauli_rotation_and_survives_the_round_trip() {
    let qasm = "OPENQASM 3.0;\ninput float[64] t;\nqubit[3] q;\nh q[0];\nrxyz(0.0) q[0], q[1], q[2];\nrxx(t) q[0], q[1];\n";
    let (template, params) = openqasm::parse_parametric(qasm).expect("parse");
    assert_eq!(params.num_slots(), 1);
    assert_eq!(params.links().len(), 1);

    let bound = params.bind(&template, &[0.63]).expect("bind");
    assert!(
        matches!(
            &bound.instructions[2],
            Instruction::Gate { gate: Gate::PauliRot(data), .. } if data.theta() == 0.63
        ),
        "expected the bound angle on a PauliRot, got {:?}",
        bound.instructions[2]
    );

    let round = round_trip(&bound);
    assert_streams_match(&bound, &round, "bound_pauli_rotation");
    let native = round
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::PauliRot(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(native, 2, "the round trip lost a native rotation");
}

// ---- Generated-program round trip ----

// A seeded source generator over the statement grammar: both headers and
// register spellings, every subscript shape, modifier chains, a gate
// definition, classical constants in angles and indices, both measure
// spellings, every condition shape with an else arm, loops over ranges and
// sets, switch and barrier. Each program is parsed, exported, parsed
// again, and the two streams and their statevector probabilities compared, so
// a construct the parser reads differently from what the exporter writes for
// it fails here with the program text in the message.
struct ProgramGen {
    state: u64,
}

impl ProgramGen {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }

    fn angle(&mut self, has_theta: bool) -> String {
        let mut choices = vec![
            "pi / 4",
            "-pi / 3",
            "0.7",
            "2 * pi / 7",
            "1e-1",
            "pi",
            "(pi + 1) / 2",
            "-0.25",
        ];
        if has_theta {
            choices.push("theta");
            choices.push("2 * theta");
        }
        self.pick(&choices).to_string()
    }

    fn qubit(&mut self, n: usize, has_k: bool) -> String {
        if has_k && self.chance(15) {
            "q[k]".to_string()
        } else {
            format!("q[{}]", self.below(n))
        }
    }

    // One or more qubits: single, range, stepped range, open range, set, or
    // the whole register.
    fn qubit_group(&mut self, n: usize) -> String {
        match self.below(6) {
            0 => "q".to_string(),
            1 => format!("q[{}]", self.below(n)),
            2 => {
                let a = self.below(n);
                let b = a + self.below(n - a);
                format!("q[{a}:{b}]")
            }
            3 if n >= 3 => format!("q[0:2:{}]", n - 1),
            4 => format!("q[{}:]", self.below(n)),
            _ => {
                let a = self.below(n);
                let b = self.below(n);
                if a == b {
                    format!("q[{{{a}}}]")
                } else {
                    format!("q[{{{a}, {b}}}]")
                }
            }
        }
    }

    fn distinct_pair(&mut self, n: usize) -> (usize, usize) {
        let a = self.below(n);
        let b = (a + 1 + self.below(n - 1)) % n;
        (a, b)
    }

    fn condition(&mut self, m: usize) -> String {
        let j = self.below(m);
        match self.below(6) {
            0 => format!("c == {}", self.below(1 << m.min(4))),
            1 => format!("c != {}", self.below(1 << m.min(4))),
            2 => format!("c[{j}]"),
            3 => format!("!c[{j}]"),
            4 if m >= 2 => format!("c[0] ^ c[{}]", m - 1),
            _ => format!("c[{j}] == {}", self.below(2)),
        }
    }

    fn program(&mut self) -> String {
        let legacy = self.chance(25);
        let n = 1 + self.below(5);
        let m = self.below(n + 1);
        let has_theta = !legacy && self.chance(40);
        let has_k = !legacy && n >= 2 && self.chance(40);
        let has_def = !legacy && n >= 2 && self.chance(40);
        let mut out = String::new();
        if legacy {
            out.push_str("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n");
            out.push_str(&format!("qreg q[{n}];\n"));
            if m > 0 {
                out.push_str(&format!("creg c[{m}];\n"));
            }
        } else {
            out.push_str("OPENQASM 3.0;\n");
            if self.chance(50) {
                out.push_str("include \"stdgates.inc\";\n");
            }
            out.push_str(&format!("qubit[{n}] q;\n"));
            if m > 0 {
                out.push_str(&format!("bit[{m}] c;\n"));
            }
        }
        if has_theta {
            out.push_str("float theta = 0.3;\n");
        }
        if has_k {
            out.push_str(&format!("const int k = {};\n", self.below(n)));
        }
        if has_def {
            out.push_str("gate rzx(t) a, b { h b; cx a, b; rz(t) b; cx a, b; h b; }\n");
        }
        let mut measured = false;
        let count = 3 + self.below(10);
        for _ in 0..count {
            match self.below(14) {
                0 | 1 => {
                    let g = self.pick(&["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx"]);
                    let target = self.qubit_group(n);
                    out.push_str(&format!("{g} {target};\n"));
                }
                2 | 3 => {
                    let g = self.pick(&["rx", "ry", "rz", "p"]);
                    let angle = self.angle(has_theta);
                    let target = self.qubit(n, has_k);
                    out.push_str(&format!("{g}({angle}) {target};\n"));
                }
                4 if n >= 2 => {
                    let (a, b) = self.distinct_pair(n);
                    let stmt = match self.below(6) {
                        0 => format!("cx q[{a}], q[{b}];"),
                        1 => format!("cz q[{a}], q[{b}];"),
                        2 => format!("swap q[{a}], q[{b}];"),
                        3 => format!("rzz({}) q[{a}], q[{b}];", self.angle(has_theta)),
                        4 => format!("crx({}) q[{a}], q[{b}];", self.angle(has_theta)),
                        _ => format!("cp({}) q[{a}], q[{b}];", self.angle(has_theta)),
                    };
                    out.push_str(&stmt);
                    out.push('\n');
                }
                5 if !legacy => {
                    let modifier =
                        self.pick(&["inv @ ", "pow(2) @ ", "pow(3) @ ", "inv @ pow(2) @ "]);
                    let g = self.pick(&["x", "y", "s", "t", "sx", "h"]);
                    let target = self.qubit(n, has_k);
                    out.push_str(&format!("{modifier}{g} {target};\n"));
                }
                6 if !legacy && n >= 2 => {
                    let (a, b) = self.distinct_pair(n);
                    let modifier = self.pick(&["ctrl @ ", "negctrl @ ", "ctrl @ inv @ "]);
                    let g = self.pick(&["x", "y", "z", "h", "s"]);
                    out.push_str(&format!("{modifier}{g} q[{a}], q[{b}];\n"));
                }
                7 if has_def => {
                    let (a, b) = self.distinct_pair(n);
                    let angle = self.angle(has_theta);
                    out.push_str(&format!("rzx({angle}) q[{a}], q[{b}];\n"));
                }
                8 if m > 0 => {
                    let i = self.below(n);
                    let j = self.below(m);
                    if legacy || self.chance(50) {
                        out.push_str(&format!("measure q[{i}] -> c[{j}];\n"));
                    } else {
                        out.push_str(&format!("c[{j}] = measure q[{i}];\n"));
                    }
                    measured = true;
                }
                9 if measured => {
                    let cond = if legacy {
                        format!("c == {}", self.below(1 << m.min(4)))
                    } else {
                        self.condition(m)
                    };
                    let g = self.pick(&["x", "z", "h"]);
                    let target = self.below(n);
                    if legacy || self.chance(50) {
                        out.push_str(&format!("if ({cond}) {g} q[{target}];\n"));
                    } else {
                        let other = self.pick(&["y", "s"]);
                        out.push_str(&format!(
                            "if ({cond}) {{ {g} q[{target}]; }} else {{ {other} q[{target}]; }}\n"
                        ));
                    }
                }
                10 if !legacy => {
                    let stmt = match self.below(3) {
                        0 => format!("for int i in [0:{}] {{ h q[i]; }}", n - 1),
                        1 if n >= 3 => format!("for int i in [0:2:{}] {{ x q[i]; }}", n - 1),
                        _ => format!("for i in {{0, {}}} {{ z q[i]; }}", n - 1),
                    };
                    out.push_str(&stmt);
                    out.push('\n');
                }
                11 if !legacy && measured => {
                    let a = self.below(n);
                    out.push_str(&format!(
                        "switch (c) {{ case 0 {{ x q[{a}]; }} case 1, 2 {{ z q[{a}]; }} default {{ h q[{a}]; }} }}\n"
                    ));
                }
                12 => {
                    let target = self.qubit_group(n);
                    out.push_str(&format!("barrier {target};\n"));
                }
                13 => {
                    out.push_str(&format!("reset q[{}];\n", self.below(n)));
                }
                _ => {
                    let target = self.qubit_group(n);
                    out.push_str(&format!("h {target};\n"));
                }
            }
        }
        out
    }
}

// A modifier chain can evaluate to a matrix payload the exporter recognizes
// as a named gate (`inv @ pow(2) @ s` is `p(-pi)`), so the stream comes back
// with a different variant carrying the same unitary. Streams match either
// exactly or, for a single-qubit gate on the same target, up to global phase.
fn assert_streams_equivalent(original: &Circuit, round: &Circuit, label: &str) {
    assert_eq!(
        original.num_qubits, round.num_qubits,
        "{label}: qubit count"
    );
    assert_eq!(
        original.num_classical_bits, round.num_classical_bits,
        "{label}: classical bit count"
    );
    assert_eq!(
        original.instructions.len(),
        round.instructions.len(),
        "{label}: instruction count"
    );
    for (i, (a, b)) in original
        .instructions
        .iter()
        .zip(&round.instructions)
        .enumerate()
    {
        if instructions_match(a, b) {
            continue;
        }
        let same_1q = match (a, b) {
            (
                Instruction::Gate {
                    gate: ga,
                    targets: ta,
                },
                Instruction::Gate {
                    gate: gb,
                    targets: tb,
                },
            ) => {
                ta == tb
                    && ga.num_qubits() == 1
                    && gb.num_qubits() == 1
                    && matrices_match_up_to_phase(&ga.matrix_2x2(), &gb.matrix_2x2())
            }
            _ => false,
        };
        assert!(
            same_1q,
            "{label}: instruction {i} differs\n  before: {a:?}\n  after:  {b:?}"
        );
    }
}

fn matrices_match_up_to_phase(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> bool {
    let (r, c) = (0..2)
        .flat_map(|r| (0..2).map(move |c| (r, c)))
        .max_by(|x, y| a[x.0][x.1].norm().total_cmp(&a[y.0][y.1].norm()))
        .expect("four entries");
    if a[r][c].norm() < PAYLOAD_EPS || b[r][c].norm() < PAYLOAD_EPS {
        return false;
    }
    let phase = b[r][c] / a[r][c];
    if (phase.norm() - 1.0).abs() > PAYLOAD_EPS {
        return false;
    }
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(x, y)| (x * phase - y).norm() < PAYLOAD_EPS)
}

#[test]
fn export_round_trips_generated_programs() {
    for seed in 0..1000u64 {
        let source = ProgramGen::new(seed).program();
        let circuit = openqasm::parse(&source)
            .unwrap_or_else(|err| panic!("seed {seed}: parse failed: {err}\n{source}"));
        let qasm = to_qasm3(&circuit)
            .unwrap_or_else(|err| panic!("seed {seed}: export failed: {err}\n{source}"));
        let round = openqasm::parse(&qasm).unwrap_or_else(|err| {
            panic!("seed {seed}: reparse failed: {err}\n{source}\n---\n{qasm}")
        });
        let label = format!("seed {seed}\n{source}\n---\n{qasm}");
        assert_streams_equivalent(&circuit, &round, &label);
        assert_probs_close(
            &sv_reference_probs(&round),
            &sv_reference_probs(&circuit),
            SV_EPS,
            &label,
        );
    }
}
