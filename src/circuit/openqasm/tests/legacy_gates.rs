//! Gates from the OpenQASM 2 `qelib1.inc` that are built in by name.

use super::*;

fn unitary_columns(source: &str) -> Vec<Vec<num_complex::Complex64>> {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;

    let circuit = parse(source).unwrap_or_else(|e| panic!("`{source}`: {e}"));
    let dim = 1usize << circuit.num_qubits;
    (0..dim)
        .map(|col| {
            let mut basis = vec![num_complex::Complex64::new(0.0, 0.0); dim];
            basis[col] = num_complex::Complex64::new(1.0, 0.0);
            let mut sv = StatevectorBackend::new(42);
            sv.init_from_amplitudes(basis, 0).unwrap();
            sv.apply_instructions(&circuit.instructions).unwrap();
            sv.export_statevector().unwrap()
        })
        .collect()
}

const QELIB1_LEGACY_GATES: &str = r#"
    gate u0_ref(gamma) q { U(0,0,0) q; }
    gate cu1_ref(lambda) a,b { u1(lambda/2) a; cx a,b; u1(-lambda/2) b; cx a,b; u1(lambda/2) b; }
    gate cu3_ref(theta,phi,lambda) c,t {
        u1((lambda+phi)/2) c;
        u1((lambda-phi)/2) t;
        cx c,t;
        u3(-theta/2,0,-(phi+lambda)/2) t;
        cx c,t;
        u3(theta/2,phi,0) t;
    }
    gate c3sqrtx_ref a,b,c,d {
        h d; cu1_ref(pi/8) a,d; h d;
        cx a,b;
        h d; cu1_ref(-pi/8) b,d; h d;
        cx a,b;
        h d; cu1_ref(pi/8) b,d; h d;
        cx b,c;
        h d; cu1_ref(-pi/8) c,d; h d;
        cx a,c;
        h d; cu1_ref(pi/8) c,d; h d;
        cx b,c;
        h d; cu1_ref(-pi/8) c,d; h d;
        cx a,c;
        h d; cu1_ref(pi/8) c,d; h d;
    }
"#;

#[test]
fn qelib1_legacy_gates_match_their_qelib1_definitions() {
    let cases = [
        (2, "cu1(pi/2) q[0],q[1];", "cu1_ref(pi/2) q[0],q[1];"),
        (2, "cu1(-0.9) q[1],q[0];", "cu1_ref(-0.9) q[1],q[0];"),
        (
            2,
            "cu3(0.7,-1.3,2.1) q[0],q[1];",
            "cu3_ref(0.7,-1.3,2.1) q[0],q[1];",
        ),
        (
            2,
            "cu3(2.5,0.4,-0.8) q[1],q[0];",
            "cu3_ref(2.5,0.4,-0.8) q[1],q[0];",
        ),
        (1, "u0(0.4) q[0];", "u0_ref(0.4) q[0];"),
        (
            4,
            "c3sqrtx q[0],q[1],q[2],q[3];",
            "c3sqrtx_ref q[0],q[1],q[2],q[3];",
        ),
        (
            4,
            "c3sqrtx q[2],q[0],q[3],q[1];",
            "c3sqrtx_ref q[2],q[0],q[3],q[1];",
        ),
        (
            3,
            "ctrl @ cu3(0.7,-1.3,2.1) q[2],q[0],q[1];",
            "ctrl @ cu3_ref(0.7,-1.3,2.1) q[2],q[0],q[1];",
        ),
        (
            2,
            "inv @ cu3(0.7,-1.3,2.1) q[0],q[1];",
            "inv @ cu3_ref(0.7,-1.3,2.1) q[0],q[1];",
        ),
        (
            3,
            "ctrl @ cu1(1.1) q[2],q[0],q[1];",
            "ctrl @ cu1_ref(1.1) q[2],q[0],q[1];",
        ),
        (
            2,
            "ctrl @ u0(0.4) q[1],q[0];",
            "ctrl @ u0_ref(0.4) q[1],q[0];",
        ),
        (
            4,
            "inv @ c3sqrtx q[0],q[1],q[2],q[3];",
            "inv @ c3sqrtx_ref q[0],q[1],q[2],q[3];",
        ),
    ];
    for (width, call, reference) in cases {
        let program = |body: &str| {
            format!(
                "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{width}];\n\
                 {QELIB1_LEGACY_GATES}\n{body}"
            )
        };
        let got = unitary_columns(&program(call));
        let want = unitary_columns(&program(reference));
        for (got, want) in got.iter().zip(&want) {
            crate::test_support::assert_amps_close(got, want, 1e-12);
        }
    }
}

#[test]
fn cu1_is_a_controlled_phase() {
    let c =
        parse("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncu1(pi/2) q[0],q[1];").unwrap();
    assert_eq!(c.instructions.len(), 1);
    let Instruction::Gate { gate, targets } = &c.instructions[0] else {
        panic!("expected a gate, got {:?}", c.instructions[0]);
    };
    assert_eq!(targets.as_slice(), &[0, 1]);
    let phase = gate.controlled_phase().expect("a controlled phase");
    assert!((phase - num_complex::Complex64::new(0.0, 1.0)).norm() < 1e-12);
}
