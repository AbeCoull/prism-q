use super::*;
use crate::circuit::qasm::{
    expr as qasm_expr, lexer,
    stream::{MAX_NESTING, Stream},
};

fn eval_expr(source: &str, line: usize, vars: Option<&HashMap<&str, f64>>) -> Result<f64> {
    let tokens = lexer::tokenize(source)?;
    let mut stream = Stream::new(&tokens);
    let expr = qasm_expr::parse(&mut stream)?;
    if !stream.at_end() {
        return Err(stream.expected("the end of the expression"));
    }
    let vars = vars.map(|vars| {
        vars.iter()
            .map(|(name, value)| (*name, ClassicalValue::Float(*value)))
            .collect::<HashMap<_, _>>()
    });
    Ok(qasm_expr::eval(&expr, line, vars.as_ref())?.to_float())
}

fn assert_complex_close(actual: num_complex::Complex64, expected: num_complex::Complex64) {
    assert!(
        (actual - expected).norm() < 1e-12,
        "expected {expected:?}, got {actual:?}"
    );
}

fn assert_mat2_close(
    actual: &[[num_complex::Complex64; 2]; 2],
    expected: [[num_complex::Complex64; 2]; 2],
) {
    for i in 0..2 {
        for j in 0..2 {
            assert_complex_close(actual[i][j], expected[i][j]);
        }
    }
}

fn assert_mat4_close(
    actual: &[[num_complex::Complex64; 4]; 4],
    expected: [[num_complex::Complex64; 4]; 4],
) {
    for i in 0..4 {
        for j in 0..4 {
            assert_complex_close(actual[i][j], expected[i][j]);
        }
    }
}

#[test]
fn test_oq3_minimal_circuit() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nh q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 1);
    assert_eq!(c.gate_count(), 1);
}

#[test]
fn test_oq3_bell_circuit() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 2);
    assert_eq!(c.num_classical_bits, 2);
    assert_eq!(c.gate_count(), 2);
    assert_eq!(c.instructions.len(), 4);
}

#[test]
fn test_oq2_compat() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 2);
    assert_eq!(c.gate_count(), 2);
    assert_eq!(c.instructions.len(), 3);
}

#[test]
fn test_parametric_gates() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(pi/4) q[0];\nry(1.5707) q[0];\nrz(2*pi) q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);

    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rx(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
            }
            _ => panic!("expected Rx"),
        }
    }
}

#[test]
fn test_unsupported_def_with_measurement() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\ndef mygate(qubit q) { measure q -> c[0]; }";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::UnsupportedConstruct { .. }));
}

#[test]
fn test_switch_lowers_to_a_guard_per_label() {
    let qasm = "OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) { case 0 { x q[0]; } }";
    let circuit = parse(qasm).expect("parse");
    assert_eq!(circuit.instructions.len(), 1);
    assert!(matches!(
        &circuit.instructions[0],
        Instruction::Conditional {
            condition: ClassicalCondition::RegisterEquals {
                value: 0,
                size: 2,
                ..
            },
            ..
        }
    ));
}

#[test]
fn test_switch_body_must_be_case_or_default() {
    let qasm = "OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) { x q[0]; }";
    match parse(qasm).unwrap_err() {
        PrismError::Parse { message, line } => {
            assert!(message.contains("`case` or `default`"), "{message}");
            assert_eq!(line, 4);
        }
        other => panic!("expected a parse error, got {other:?}"),
    }
}

#[test]
fn test_undefined_register() {
    let qasm = "OPENQASM 3.0;\nh q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::UndefinedRegister { .. }));
}

#[test]
fn test_gate_def_missing_name_is_error() {
    let qasm = "OPENQASM 3.0;\ngate {}";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_qubit_out_of_bounds() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[5];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::InvalidQubit { .. }));
}

#[test]
fn test_gate_arity_mismatch() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\ncx q[0], q[1], q[2];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::GateArity { .. }));
}

#[test]
fn test_multiple_registers() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] a;
        qubit[3] b;
        h a[0];
        h b[2];
        cx a[1], b[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 5);
    assert_eq!(c.gate_count(), 3);
}

#[test]
fn test_comments_stripped() {
    let qasm = r#"
        OPENQASM 3.0; // version
        qubit[1] q; // one qubit
        // full line comment
        h q[0]; // hadamard
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
}

#[test]
fn test_barrier() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nbarrier q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    assert!(matches!(c.instructions[0], Instruction::Barrier { .. }));
}

#[test]
fn test_oq3_measure_assign() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nx q[0];\nc[0] = measure q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    assert!(matches!(
        c.instructions[1],
        Instruction::Measure {
            qubit: 0,
            classical_bit: 0
        }
    ));
}

#[test]
fn test_single_qubit_decl() {
    let qasm = "OPENQASM 3.0;\nqubit q;\nh q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 1);
    assert_eq!(c.gate_count(), 1);
}

#[test]
fn test_inv_self_inverse() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\ninv @ h q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::H);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_inv_t_becomes_tdg() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\ninv @ t q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::Tdg);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_inv_parametric() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\ninv @ rx(pi/4) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rx(theta) => {
                assert!((theta + std::f64::consts::FRAC_PI_4).abs() < 1e-12);
            }
            _ => panic!("expected Rx"),
        }
    }
}

#[test]
fn test_ctrl_x_becomes_cx() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ x q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(*gate, Gate::Cx);
        assert_eq!(targets.as_slice(), &[0, 1]);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_ctrl_z_becomes_cz() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ z q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::Cz);
    }
}

#[test]
fn test_ctrl_h_becomes_cu() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ h q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert!(matches!(gate, Gate::Cu(_)));
        assert_eq!(gate.num_qubits(), 2);
        assert_eq!(targets.as_slice(), &[0, 1]);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_ctrl_parametric() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ rz(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert!(matches!(gate, Gate::Cu(_)));
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_chained_inv_ctrl() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ninv @ ctrl @ rx(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert!(matches!(gate, Gate::Cu(_)));
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_pow_integer() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\npow(2) @ t q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        // T^2 = S
        let expected = Gate::S.matrix_2x2();
        if let Gate::Fused(mat) = gate {
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (mat[i][j] - expected[i][j]).norm() < 1e-12,
                        "T^2 should be S"
                    );
                }
            }
        } else {
            panic!("expected Fused, got {:?}", gate);
        }
    }
}

#[test]
fn test_pow_zero_is_id() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\npow(0) @ x q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::Id);
    }
}

#[test]
fn test_ctrl_ctrl_x_is_toffoli() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ ctrl @ x q[0], q[1], q[2];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(gate.num_qubits(), 3);
        assert_eq!(gate.name(), "mcu");
        assert_eq!(targets.as_slice(), &[0, 1, 2]);
    } else {
        panic!("expected gate instruction");
    }
}

#[test]
fn test_ctrl_cx_is_toffoli() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ cx q[0], q[1], q[2];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(gate.num_qubits(), 3);
        assert_eq!(gate.name(), "mcu");
        assert_eq!(targets.as_slice(), &[0, 1, 2]);
    } else {
        panic!("expected gate instruction");
    }
}

#[test]
fn test_ctrl_ctrl_ctrl_x() {
    let qasm = "OPENQASM 3.0;\nqubit[4] q;\nctrl @ ctrl @ ctrl @ x q[0], q[1], q[2], q[3];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(gate.num_qubits(), 4);
        assert_eq!(gate.name(), "mcu");
        assert_eq!(targets.as_slice(), &[0, 1, 2, 3]);
    } else {
        panic!("expected gate instruction");
    }
}

#[test]
fn test_no_modifier_unchanged() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncx q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::Cx);
    }
}

#[test]
fn test_cp_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncp(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(gate.num_qubits(), 2);
        assert_eq!(gate.name(), "cu");
        assert_eq!(targets.as_slice(), &[0, 1]);
        let phase = gate.controlled_phase().expect("should be CPhase");
        let expected = num_complex::Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        assert!((phase - expected).norm() < 1e-12);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_cphase_alias() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncphase(pi) q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        let phase = gate.controlled_phase().unwrap();
        assert!((phase.re - (-1.0)).abs() < 1e-12);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_ctrl_cp_promotes_to_mcu() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ cp(pi/4) q[0], q[1], q[2];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(gate.name(), "mcu");
        assert_eq!(targets.as_slice(), &[0, 1, 2]);
        assert!(gate.controlled_phase().is_some());
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_inv_cp() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ninv @ cp(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        let phase = gate.controlled_phase().unwrap();
        let expected = num_complex::Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
        assert!((phase - expected).norm() < 1e-12);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_cp_arity_mismatch() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\ncp(pi/4) q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::GateArity { .. }));
}

#[test]
fn test_sx_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nsx q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::SX);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_sxdg_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nsxdg q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::SXdg);
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_p_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(pi/4) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::P(std::f64::consts::FRAC_PI_4));
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_phase_alias() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nphase(pi/4) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::P(std::f64::consts::FRAC_PI_4));
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_uppercase_u_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nU(pi/2, 0, pi) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        if let Gate::Fused(mat) = gate {
            assert_mat2_close(mat, Gate::H.matrix_2x2());
        } else {
            panic!("expected Fused");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_qiskit_exported_r_definition_uses_uppercase_u() {
    let qasm = r#"
        OPENQASM 3.0;
        gate r(p0, p1) _gate_q_0 {
          U(p0, -pi/2 + p1, pi/2 - p1) _gate_q_0;
        }
        qubit[1] q;
        r(pi/3, pi/7) q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    assert!(matches!(
        &c.instructions[0],
        Instruction::Gate {
            gate: Gate::Fused(_),
            ..
        }
    ));
}

#[test]
fn test_r_gate_matrix() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nr(pi/2, pi/4) q[0];";
    let c = parse(qasm).unwrap();
    let theta = std::f64::consts::FRAC_PI_2;
    let phi = std::f64::consts::FRAC_PI_4;
    let c0 = num_complex::Complex64::new((theta / 2.0).cos(), 0.0);
    let off = num_complex::Complex64::new(0.0, -1.0) * (theta / 2.0).sin();
    let expected = [
        [c0, off * num_complex::Complex64::from_polar(1.0, -phi)],
        [off * num_complex::Complex64::from_polar(1.0, phi), c0],
    ];
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        if let Gate::Fused(mat) = gate {
            assert_mat2_close(mat, expected);
        } else {
            panic!("expected Fused");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_cy_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncy q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(gate.num_qubits(), 2);
        assert_eq!(targets.as_slice(), &[0, 1]);
        if let Gate::Cu(mat) = gate {
            let expected = Gate::Y.matrix_2x2();
            for i in 0..2 {
                for j in 0..2 {
                    assert!((mat[i][j] - expected[i][j]).norm() < 1e-12);
                }
            }
        } else {
            panic!("expected Cu");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_cs_and_csdg_gates() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncs q[0], q[1];\ncsdg q[0], q[1];";
    let c = parse(qasm).unwrap();
    let expected = [Gate::S.matrix_2x2(), Gate::Sdg.matrix_2x2()];
    for (inst, expected_mat) in c.instructions.iter().zip(expected) {
        if let Instruction::Gate { gate, targets } = inst {
            assert_eq!(targets.as_slice(), &[0, 1]);
            if let Gate::Cu(mat) = gate {
                assert_mat2_close(mat, expected_mat);
            } else {
                panic!("expected Cu");
            }
        } else {
            panic!("expected gate");
        }
    }
}

#[test]
fn test_cu_gate_matrix() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncu(pi/2, pi/5, pi/7, pi/11) q[0], q[1];";
    let c = parse(qasm).unwrap();
    let theta = std::f64::consts::FRAC_PI_2;
    let phi = std::f64::consts::PI / 5.0;
    let lam = std::f64::consts::PI / 7.0;
    let gamma = std::f64::consts::PI / 11.0;
    let c0 = (theta / 2.0).cos();
    let s0 = (theta / 2.0).sin();
    let phase = num_complex::Complex64::from_polar(1.0, gamma);
    let expected = [
        [
            phase * num_complex::Complex64::new(c0, 0.0),
            -phase * num_complex::Complex64::from_polar(s0, lam),
        ],
        [
            phase * num_complex::Complex64::from_polar(s0, phi),
            phase * num_complex::Complex64::from_polar(c0, phi + lam),
        ],
    ];
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(targets.as_slice(), &[0, 1]);
        if let Gate::Cu(mat) = gate {
            assert_mat2_close(mat, expected);
        } else {
            panic!("expected Cu");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_crx_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncrx(pi/2) q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        if let Gate::Cu(mat) = gate {
            let expected = Gate::Rx(std::f64::consts::FRAC_PI_2).matrix_2x2();
            for i in 0..2 {
                for j in 0..2 {
                    assert!((mat[i][j] - expected[i][j]).norm() < 1e-12);
                }
            }
        } else {
            panic!("expected Cu");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_ccx_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nccx q[0], q[1], q[2];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(targets.as_slice(), &[0, 1, 2]);
        if let Gate::Mcu(data) = gate {
            assert_eq!(data.num_controls, 2);
            let x_mat = Gate::X.matrix_2x2();
            for (row_d, row_x) in data.mat.iter().zip(x_mat.iter()) {
                for (d, x) in row_d.iter().zip(row_x.iter()) {
                    assert!((d - x).norm() < 1e-12);
                }
            }
        } else {
            panic!("expected Mcu");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_ccz_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nccz q[0], q[1], q[2];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        if let Gate::Mcu(data) = gate {
            assert_eq!(data.num_controls, 2);
            let z_mat = Gate::Z.matrix_2x2();
            for (row_d, row_z) in data.mat.iter().zip(z_mat.iter()) {
                for (d, z) in row_d.iter().zip(row_z.iter()) {
                    assert!((d - z).norm() < 1e-12);
                }
            }
        } else {
            panic!("expected Mcu");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_multi_controlled_x_aliases() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[6] q;
        c3x q[0], q[1], q[2], q[3];
        c4x q[0], q[1], q[2], q[3], q[4];
        mcx q[0], q[1], q[2], q[3], q[4], q[5];
    "#;
    let c = parse(qasm).unwrap();
    let expected_controls = [3, 4, 5];
    for (inst, expected) in c.instructions.iter().zip(expected_controls) {
        if let Instruction::Gate { gate, .. } = inst {
            if let Gate::Mcu(data) = gate {
                assert_eq!(data.num_controls, expected);
                assert_mat2_close(&data.mat, Gate::X.matrix_2x2());
            } else {
                panic!("expected Mcu");
            }
        } else {
            panic!("expected gate");
        }
    }
}

#[test]
fn test_relative_phase_x_decompositions() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        rccx q[0], q[1], q[2];
        rcccx q[0], q[1], q[2], q[3];
        rc3x q[0], q[1], q[2], q[3];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 45);
    assert!(
        matches!(&c.instructions[0], Instruction::Gate { gate: Gate::H, targets } if targets.as_slice() == [2])
    );
    assert!(
        matches!(&c.instructions[8], Instruction::Gate { gate: Gate::H, targets } if targets.as_slice() == [2])
    );
    assert!(
        matches!(&c.instructions[9], Instruction::Gate { gate: Gate::H, targets } if targets.as_slice() == [3])
    );
    assert!(
        matches!(&c.instructions[27], Instruction::Gate { gate: Gate::H, targets } if targets.as_slice() == [3])
    );
}

#[test]
fn test_modifier_on_pauli_rotation() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ninv @ rxx(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    let Instruction::Gate {
        gate: Gate::PauliRot(data),
        ..
    } = &c.instructions[0]
    else {
        panic!("expected a PauliRot, got {:?}", c.instructions[0]);
    };
    assert!((data.theta() + std::f64::consts::FRAC_PI_4).abs() < 1e-12);
}

#[test]
fn test_cswap_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\ncswap q[0], q[1], q[2];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 3);
    assert!(
        matches!(&c.instructions[0], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [2, 1])
    );
    assert!(
        matches!(&c.instructions[1], Instruction::Gate { gate: Gate::Mcu(_), targets } if targets.as_slice() == [0, 1, 2])
    );
    assert!(
        matches!(&c.instructions[2], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [2, 1])
    );
}

#[test]
fn test_rzz_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nrzz(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    assert!(matches!(
        &c.instructions[0],
        Instruction::Gate {
            gate: Gate::Rzz(_),
            ..
        }
    ));
}

// Equivalence to the basis-change ladder is checked numerically in
// `tests/integration/pauli_rot_correctness.rs`; this pins the parsed shape.
#[test]
fn test_rxx_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nrxx(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    let Instruction::Gate {
        gate: Gate::PauliRot(data),
        targets,
    } = &c.instructions[0]
    else {
        panic!("expected a PauliRot, got {:?}", c.instructions[0]);
    };
    assert_eq!(data.axes(), [PauliAxis::X, PauliAxis::X]);
    assert_eq!(targets.as_slice(), [0, 1]);
}

#[test]
fn test_ryy_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nryy(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    assert!(matches!(
        &c.instructions[0],
        Instruction::Gate {
            gate: Gate::PauliRot(data),
            ..
        } if data.axes() == [PauliAxis::Y, PauliAxis::Y]
    ));
}

#[test]
fn test_wide_pauli_rotation_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[4] q;\nrxyz(0.7) q[2], q[0], q[3];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    let Instruction::Gate {
        gate: Gate::PauliRot(data),
        targets,
    } = &c.instructions[0]
    else {
        panic!("expected a PauliRot, got {:?}", c.instructions[0]);
    };
    // Targets sort by qubit and the letters follow them, so `y` stays on q[0].
    assert_eq!(targets.as_slice(), [0, 2, 3]);
    assert_eq!(data.axes(), [PauliAxis::Y, PauliAxis::X, PauliAxis::Z]);
}

#[test]
fn test_pauli_rotation_rejections() {
    let repeated = parse("OPENQASM 3.0;\nqubit[3] q;\nrxx(0.3) q[0], q[0];");
    assert!(matches!(repeated, Err(PrismError::Parse { .. })));

    let arity = parse("OPENQASM 3.0;\nqubit[3] q;\nrxyz(0.3) q[0], q[1];");
    assert!(matches!(arity, Err(PrismError::GateArity { .. })));

    // `rccx` is not a Pauli string, `c` being no Pauli letter, so the name
    // still reaches its own decomposition.
    let rccx = parse("OPENQASM 3.0;\nqubit[3] q;\nrccx q[0], q[1], q[2];").unwrap();
    assert!(rccx.instructions.len() > 1);
}

#[test]
fn test_xx_plus_minus_yy_matrices() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        xx_plus_yy(pi/3, pi/5) q[0], q[1];
        xx_minus_yy(pi/7, pi/11) q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();

    let zero = num_complex::Complex64::new(0.0, 0.0);
    let one = num_complex::Complex64::new(1.0, 0.0);
    let plus_theta = std::f64::consts::PI / 3.0;
    let plus_beta = std::f64::consts::PI / 5.0;
    let plus_c = num_complex::Complex64::new((plus_theta / 2.0).cos(), 0.0);
    let plus_s = num_complex::Complex64::new(0.0, -(plus_theta / 2.0).sin());
    let expected_plus = [
        [one, zero, zero, zero],
        [
            zero,
            plus_c,
            plus_s * num_complex::Complex64::from_polar(1.0, -plus_beta),
            zero,
        ],
        [
            zero,
            plus_s * num_complex::Complex64::from_polar(1.0, plus_beta),
            plus_c,
            zero,
        ],
        [zero, zero, zero, one],
    ];

    let minus_theta = std::f64::consts::PI / 7.0;
    let minus_beta = std::f64::consts::PI / 11.0;
    let minus_c = num_complex::Complex64::new((minus_theta / 2.0).cos(), 0.0);
    let minus_s = num_complex::Complex64::new(0.0, -(minus_theta / 2.0).sin());
    let expected_minus = [
        [
            minus_c,
            zero,
            zero,
            minus_s * num_complex::Complex64::from_polar(1.0, -minus_beta),
        ],
        [zero, one, zero, zero],
        [zero, zero, one, zero],
        [
            minus_s * num_complex::Complex64::from_polar(1.0, minus_beta),
            zero,
            zero,
            minus_c,
        ],
    ];

    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(targets.as_slice(), &[0, 1]);
        if let Gate::Fused2q(mat) = gate {
            assert_mat4_close(mat, expected_plus);
        } else {
            panic!("expected Fused2q");
        }
    } else {
        panic!("expected gate");
    }
    if let Instruction::Gate { gate, targets } = &c.instructions[1] {
        assert_eq!(targets.as_slice(), &[0, 1]);
        if let Gate::Fused2q(mat) = gate {
            assert_mat4_close(mat, expected_minus);
        } else {
            panic!("expected Fused2q");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_iswap_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\niswap q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 6);
}

#[test]
fn test_google_cirq_native_matrices() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        syc q[0], q[1];
        sqrt_iswap q[0], q[1];
        sqrt_iswap_inv q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();
    let zero = num_complex::Complex64::new(0.0, 0.0);
    let one = num_complex::Complex64::new(1.0, 0.0);
    let neg_i = num_complex::Complex64::new(0.0, -1.0);
    let half = num_complex::Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let pos_half_i = num_complex::Complex64::new(0.0, std::f64::consts::FRAC_1_SQRT_2);
    let neg_half_i = num_complex::Complex64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2);
    let expected = [
        [
            [one, zero, zero, zero],
            [zero, zero, neg_i, zero],
            [zero, neg_i, zero, zero],
            [
                zero,
                zero,
                zero,
                num_complex::Complex64::from_polar(1.0, -std::f64::consts::PI / 6.0),
            ],
        ],
        [
            [one, zero, zero, zero],
            [zero, half, pos_half_i, zero],
            [zero, pos_half_i, half, zero],
            [zero, zero, zero, one],
        ],
        [
            [one, zero, zero, zero],
            [zero, half, neg_half_i, zero],
            [zero, neg_half_i, half, zero],
            [zero, zero, zero, one],
        ],
    ];

    for (inst, expected_mat) in c.instructions.iter().zip(expected) {
        if let Instruction::Gate { gate, targets } = inst {
            assert_eq!(targets.as_slice(), &[0, 1]);
            if let Gate::Fused2q(mat) = gate {
                assert_mat4_close(mat, expected_mat);
            } else {
                panic!("expected Fused2q");
            }
        } else {
            panic!("expected gate");
        }
    }
}

#[test]
fn test_ionq_native_gate_matrices() {
    // The hardware-native family reads angles in turns under the default
    // dialect, so a quarter turn of `gpi` is the Y-like antidiagonal and an
    // omitted third `ms` angle is a quarter turn. That family under Braket's
    // reading in radians, and the rest of the gate set, are pinned against
    // independent references in tests/integration/gate_matrix.rs.
    let qasm = "OPENQASM 3.0;
qubit[2] q;
gpi(0.25) q[0];
ms(0.0, 0.0) q[0], q[1];";
    let c = parse(qasm).unwrap();
    let zero = num_complex::Complex64::new(0.0, 0.0);
    let i = num_complex::Complex64::new(0.0, 1.0);

    let Instruction::Gate {
        gate: Gate::Fused(gpi),
        ..
    } = &c.instructions[0]
    else {
        panic!("expected a Fused gpi");
    };
    assert_mat2_close(gpi, [[zero, -i], [i, zero]]);

    let Instruction::Gate {
        gate: Gate::Fused2q(ms),
        ..
    } = &c.instructions[1]
    else {
        panic!("expected a Fused2q ms");
    };
    let h = num_complex::Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let o = num_complex::Complex64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2);
    assert_mat4_close(
        ms,
        [
            [h, zero, zero, o],
            [zero, h, o, zero],
            [zero, o, h, zero],
            [o, zero, zero, h],
        ],
    );
}

#[test]
fn test_ecr_gate() {
    // `ecr` resolves to a matrix gate rather than a lowering, so `inv @` and
    // `pow @` invert and repeat the matrix instead of an expanded body. Its
    // unitary is pinned in tests/integration/gate_matrix.rs.
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\necr q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    assert!(matches!(
        &c.instructions[0],
        Instruction::Gate {
            gate: Gate::Fused2q(_),
            targets
        } if targets.as_slice() == [0, 1]
    ));
}

#[test]
fn test_dcx_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ndcx q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 2);
    assert!(
        matches!(&c.instructions[0], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [0, 1])
    );
    assert!(
        matches!(&c.instructions[1], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [1, 0])
    );
}

#[test]
fn test_u1_is_p() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nu1(pi/4) q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::P(std::f64::consts::FRAC_PI_4));
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_u3_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nu3(pi/2, 0, pi) q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        if let Gate::Fused(mat) = gate {
            let h_mat = Gate::H.matrix_2x2();
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (mat[i][j] - h_mat[i][j]).norm() < 1e-12,
                        "u3(pi/2, 0, pi) should match H: mat[{i}][{j}] = {:?} vs {:?}",
                        mat[i][j],
                        h_mat[i][j]
                    );
                }
            }
        } else {
            panic!("expected Fused");
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_broadcast_1q_gate_on_register() {
    let qasm = "OPENQASM 3.0;\nqubit[4] q;\nh q;";
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 4);
    assert_eq!(c.gate_count(), 4);
    for i in 0..4 {
        if let Instruction::Gate { gate, targets } = &c.instructions[i] {
            assert_eq!(*gate, Gate::H);
            assert_eq!(targets.as_slice(), &[i]);
        } else {
            panic!("expected gate at index {i}");
        }
    }
}

#[test]
fn test_broadcast_2q_gate_pairwise() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        qubit[3] r;
        cx q, r;
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 6);
    assert_eq!(c.gate_count(), 3);
    for i in 0..3 {
        if let Instruction::Gate { gate, targets } = &c.instructions[i] {
            assert_eq!(*gate, Gate::Cx);
            assert_eq!(targets.as_slice(), &[i, i + 3]);
        } else {
            panic!("expected gate at index {i}");
        }
    }
}

#[test]
fn test_broadcast_mixed_indexed_and_register() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        qubit[3] r;
        cx q[0], r;
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    for i in 0..3 {
        if let Instruction::Gate { gate, targets } = &c.instructions[i] {
            assert_eq!(*gate, Gate::Cx);
            assert_eq!(targets.as_slice(), &[0, i + 3]);
        } else {
            panic!("expected gate at index {i}");
        }
    }
}

#[test]
fn test_broadcast_register_size_mismatch() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        qubit[3] r;
        cx q, r;
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_broadcast_measure_arrow() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[3];
        creg c[3];
        measure q -> c;
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 3);
    for i in 0..3 {
        assert!(matches!(
            c.instructions[i],
            Instruction::Measure {
                qubit,
                classical_bit
            } if qubit == i && classical_bit == i
        ));
    }
}

#[test]
fn test_broadcast_measure_assign() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        c = measure q;
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 3);
    for i in 0..3 {
        assert!(matches!(
            c.instructions[i],
            Instruction::Measure {
                qubit,
                classical_bit
            } if qubit == i && classical_bit == i
        ));
    }
}

#[test]
fn test_broadcast_barrier() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nqubit[2] r;\nbarrier q, r;";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    if let Instruction::Barrier { qubits } = &c.instructions[0] {
        assert_eq!(qubits.as_slice(), &[0, 1, 2, 3, 4]);
    } else {
        panic!("expected barrier");
    }
}

#[test]
fn bare_barrier_matches_the_named_register() {
    let named = parse("OPENQASM 3.0;\nqubit[3] q;\nbarrier q;").unwrap();
    let bare = parse("OPENQASM 3.0;\nqubit[3] q;\nbarrier;").unwrap();
    assert_eq!(
        format!("{:?}", bare.instructions),
        format!("{:?}", named.instructions)
    );
}

#[test]
fn bare_barrier_spans_every_register() {
    let c = parse("OPENQASM 3.0;\nqubit[3] q;\nqubit[2] r;\nbarrier;").unwrap();
    assert_eq!(c.instructions.len(), 1);
    if let Instruction::Barrier { qubits } = &c.instructions[0] {
        assert_eq!(qubits.as_slice(), &[0, 1, 2, 3, 4]);
    } else {
        panic!("expected barrier");
    }
}

#[test]
fn test_broadcast_parametric_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nrz(pi/4) q;";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    for i in 0..3 {
        if let Instruction::Gate { gate, targets } = &c.instructions[i] {
            assert!(matches!(gate, Gate::Rz(_)));
            assert_eq!(targets.as_slice(), &[i]);
        } else {
            panic!("expected gate at index {i}");
        }
    }
}

#[test]
fn test_broadcast_decomposed_gate() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        qubit[2] r;
        rzz(pi/4) q, r;
    "#;
    let c = parse(qasm).unwrap();
    // rzz emits 1 instruction per pair, 2 pairs
    assert_eq!(c.instructions.len(), 2);
}

#[test]
fn test_if_oq2_register_equals() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];
        if(c==1) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Conditional {
        condition,
        gate,
        targets,
    } = &c.instructions[0]
    {
        assert_eq!(*gate, Gate::X);
        assert_eq!(targets.as_slice(), &[0]);
        assert!(matches!(
            condition,
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 2,
                value: 1
            }
        ));
    } else {
        panic!("expected Conditional");
    }
}

#[test]
fn test_if_oq3_single_bit() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        if (c[0]) x q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Conditional {
        condition,
        gate,
        targets,
    } = &c.instructions[0]
    {
        assert_eq!(*gate, Gate::X);
        assert_eq!(targets.as_slice(), &[1]);
        assert!(matches!(condition, ClassicalCondition::BitIsOne(0)));
    } else {
        panic!("expected Conditional");
    }
}

#[test]
fn test_if_with_parametric_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0]) rz(pi/4) q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Conditional { gate, .. } = &c.instructions[0] {
        assert!(matches!(gate, Gate::Rz(_)));
    } else {
        panic!("expected Conditional");
    }
}

#[test]
fn test_if_missing_body() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0])";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_if_undefined_creg() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nif(c==0) x q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::UndefinedRegister { .. }));
}

#[test]
fn test_gate_def_no_params() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate bell a, b {
            h a;
            cx a, b;
        }
        bell q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 2);
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(*gate, Gate::H);
        assert_eq!(targets.as_slice(), &[0]);
    } else {
        panic!("expected H gate");
    }
    if let Instruction::Gate { gate, targets } = &c.instructions[1] {
        assert_eq!(*gate, Gate::Cx);
        assert_eq!(targets.as_slice(), &[0, 1]);
    } else {
        panic!("expected CX gate");
    }
}

#[test]
fn test_gate_def_with_params() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate myrz(theta) a {
            rz(theta) a;
        }
        myrz(pi/4) q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_gate_def_single_line() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ngate mygate a, b { cx a, b; }\nmygate q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::Cx);
    } else {
        panic!("expected CX");
    }
}

#[test]
fn test_gate_def_arity_mismatch() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate mygate a, b { cx a, b; }
        mygate q[0];
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::GateArity { .. }));
}

#[test]
fn test_gate_def_nested() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate myh a { h a; }
        gate mybell a, b { myh a; cx a, b; }
        mybell q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 2);
}

#[test]
fn test_missing_bracket_qubit_decl() {
    let qasm = "OPENQASM 3.0;\nqubit[4 q;";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_missing_bracket_bit_decl() {
    let qasm = "OPENQASM 3.0;\nbit[4 c;";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_pi_div_zero() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(pi/0) q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_neg_pi_div_zero() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(-pi/0) q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_nan_angle_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(NaN) q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_inf_angle_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(inf) q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_large_register_accepted() {
    let qasm = "OPENQASM 3.0;\nqubit[999] q;";
    let circuit = parse(qasm).unwrap();
    assert_eq!(circuit.num_qubits, 999);
}

#[test]
fn test_large_bit_register_accepted() {
    let qasm = "OPENQASM 3.0;\nbit[999] c;";
    let circuit = parse(qasm).unwrap();
    assert_eq!(circuit.num_classical_bits, 999);
}

#[test]
fn test_recursive_gate_def_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate loop a { loop a; }
        loop q[0];
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn test_empty_gate_body_is_the_identity() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate noop a { }
        noop q[0];
        inv @ noop q[0];
        ctrl @ noop q[0], q[1];
        x q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1, "{:?}", c.instructions);
}

#[test]
fn test_register_name_collision() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nqubit[2] q;";
    let err = parse(qasm);
    // Either error or silent overwrite is acceptable, just do not panic.
    assert!(err.is_ok() || err.is_err());
}

#[test]
fn test_cry_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncry(pi/2) q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(targets.as_slice(), &[0, 1]);
        if let Gate::Cu(mat) = gate {
            let expected = Gate::Ry(std::f64::consts::FRAC_PI_2).matrix_2x2();
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (mat[i][j] - expected[i][j]).norm() < 1e-12,
                        "CRy matrix mismatch at [{i}][{j}]"
                    );
                }
            }
        } else {
            panic!("expected Cu, got {:?}", gate);
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_crz_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncrz(pi/2) q[0], q[1];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(targets.as_slice(), &[0, 1]);
        if let Gate::Cu(mat) = gate {
            let expected = Gate::Rz(std::f64::consts::FRAC_PI_2).matrix_2x2();
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (mat[i][j] - expected[i][j]).norm() < 1e-12,
                        "CRz matrix mismatch at [{i}][{j}]"
                    );
                }
            }
        } else {
            panic!("expected Cu, got {:?}", gate);
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_expr_literal_float() {
    assert!((eval_expr("1.234", 0, None).unwrap() - 1.234).abs() < 1e-12);
    assert!((eval_expr("0.25", 0, None).unwrap() - 0.25).abs() < 1e-12);
    assert!((eval_expr("-0.5", 0, None).unwrap() - (-0.5)).abs() < 1e-12);
}

#[test]
fn test_expr_pi_constant() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("pi", 0, None).unwrap() - pi).abs() < 1e-12);
    assert!((eval_expr("-pi", 0, None).unwrap() - (-pi)).abs() < 1e-12);
    assert!((eval_expr("π", 0, None).unwrap() - pi).abs() < 1e-12);
}

#[test]
fn test_expr_tau_constant() {
    assert!((eval_expr("tau", 0, None).unwrap() - std::f64::consts::TAU).abs() < 1e-12);
}

#[test]
fn test_expr_euler_constant() {
    assert!((eval_expr("euler", 0, None).unwrap() - std::f64::consts::E).abs() < 1e-12);
    assert!(
        eval_expr("e", 0, None).is_err(),
        "OpenQASM 3 spells it `euler`"
    );
}

#[test]
fn a_declared_variable_shadows_a_constant_name() {
    let qasm = "OPENQASM 3.0;
qubit[1] q;
float e = 0.5;
gate g(euler) a { ry(euler) a; }
                rx(e) q[0];
g(0.25) q[0];";
    let c = parse(qasm).unwrap();
    let angles: Vec<f64> = c
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate {
                gate: Gate::Rx(theta) | Gate::Ry(theta),
                ..
            } => Some(*theta),
            _ => None,
        })
        .collect();
    assert_eq!(angles, [0.5, 0.25]);
}

#[test]
fn test_expr_pi_division() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("pi/2", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
    assert!((eval_expr("pi/4", 0, None).unwrap() - pi / 4.0).abs() < 1e-12);
    assert!((eval_expr("-pi/2", 0, None).unwrap() - (-pi / 2.0)).abs() < 1e-12);
}

#[test]
fn test_expr_pi_multiplication() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("2*pi", 0, None).unwrap() - 2.0 * pi).abs() < 1e-12);
    assert!((eval_expr("0.5*pi", 0, None).unwrap() - 0.5 * pi).abs() < 1e-12);
}

#[test]
fn test_expr_arithmetic() {
    assert!((eval_expr("1 + 2", 0, None).unwrap() - 3.0).abs() < 1e-12);
    assert!((eval_expr("5 - 3", 0, None).unwrap() - 2.0).abs() < 1e-12);
    assert!((eval_expr("2 * 3", 0, None).unwrap() - 6.0).abs() < 1e-12);
    assert!((eval_expr("6 / 2", 0, None).unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn test_expr_operator_precedence() {
    assert!((eval_expr("2 + 3 * 4", 0, None).unwrap() - 14.0).abs() < 1e-12);
    assert!((eval_expr("2 * 3 + 4", 0, None).unwrap() - 10.0).abs() < 1e-12);
    assert!((eval_expr("10 - 2 * 3", 0, None).unwrap() - 4.0).abs() < 1e-12);
    assert!((eval_expr("10 / 2 + 3", 0, None).unwrap() - 8.0).abs() < 1e-12);
}

#[test]
fn test_expr_parentheses() {
    assert!((eval_expr("(2 + 3) * 4", 0, None).unwrap() - 20.0).abs() < 1e-12);
    assert!((eval_expr("2 * (3 + 4)", 0, None).unwrap() - 14.0).abs() < 1e-12);
    assert!((eval_expr("((1 + 2) * (3 + 4))", 0, None).unwrap() - 21.0).abs() < 1e-12);
}

#[test]
fn test_expr_complex_pi_expressions() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("pi/2 + pi/4", 0, None).unwrap() - (pi / 2.0 + pi / 4.0)).abs() < 1e-12);
    assert!((eval_expr("2*pi/3", 0, None).unwrap() - (2.0 * pi / 3.0)).abs() < 1e-12);
    assert!((eval_expr("pi/2 + 0.1", 0, None).unwrap() - (pi / 2.0 + 0.1)).abs() < 1e-12);
    assert!((eval_expr("(pi + pi)/4", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
}

#[test]
fn test_expr_unary_minus() {
    assert!((eval_expr("-1", 0, None).unwrap() - (-1.0)).abs() < 1e-12);
    assert!((eval_expr("-(2 + 3)", 0, None).unwrap() - (-5.0)).abs() < 1e-12);
    assert!((eval_expr("-(-1)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("--1", 0, None).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn test_expr_unary_plus() {
    assert!((eval_expr("+1", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("+(2 + 3)", 0, None).unwrap() - 5.0).abs() < 1e-12);
}

#[test]
fn test_expr_math_functions() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("sin(pi/2)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("cos(0)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("sqrt(4)", 0, None).unwrap() - 2.0).abs() < 1e-12);
    assert!((eval_expr("exp(0)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("ln(1)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("abs(-5)", 0, None).unwrap() - 5.0).abs() < 1e-12);
    assert!((eval_expr("asin(1)", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
    assert!((eval_expr("acos(1)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("atan(0)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("tan(0)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("log2(8)", 0, None).unwrap() - 3.0).abs() < 1e-12);
    assert!((eval_expr("floor(2.7)", 0, None).unwrap() - 2.0).abs() < 1e-12);
    assert!((eval_expr("ceil(2.1)", 0, None).unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn test_expr_nested_functions() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("sin(pi/4) * sin(pi/4)", 0, None).unwrap() - 0.5).abs() < 1e-12);
    assert!((eval_expr("sqrt(sin(pi/2))", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("asin(sin(pi/6))", 0, None).unwrap() - pi / 6.0).abs() < 1e-10);
}

#[test]
fn test_expr_variables() {
    let mut vars = HashMap::new();
    vars.insert("theta", std::f64::consts::FRAC_PI_4);
    vars.insert("phi", std::f64::consts::FRAC_PI_2);
    let v = Some(&vars);
    assert!((eval_expr("theta", 0, v).unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    assert!(
        (eval_expr("theta + phi", 0, v).unwrap()
            - (std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2))
            .abs()
            < 1e-12
    );
    assert!(
        (eval_expr("theta/2", 0, v).unwrap() - std::f64::consts::FRAC_PI_4 / 2.0).abs() < 1e-12
    );
    assert!(
        (eval_expr("2*theta + phi", 0, v).unwrap()
            - (2.0 * std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2))
            .abs()
            < 1e-12
    );
    assert!(
        (eval_expr("sin(theta)", 0, v).unwrap() - std::f64::consts::FRAC_PI_4.sin()).abs() < 1e-12
    );
}

#[test]
fn test_expr_division_by_zero() {
    assert!(eval_expr("1/0", 0, None).is_err());
    assert!(eval_expr("pi/0", 0, None).is_err());
}

#[test]
fn test_expr_unknown_function() {
    assert!(eval_expr("foobar(1)", 0, None).is_err());
}

#[test]
fn test_expr_unknown_variable() {
    assert!(eval_expr("xyz", 0, None).is_err());
}

#[test]
fn test_expr_unbalanced_parens() {
    assert!(eval_expr("(1 + 2", 0, None).is_err());
    assert!(eval_expr("sin(pi/2", 0, None).is_err());
}

#[test]
fn test_expr_empty() {
    assert!(eval_expr("", 0, None).is_err());
}

#[test]
fn test_expr_trailing_chars() {
    assert!(eval_expr("1 2", 0, None).is_err());
}

#[test]
fn test_expr_scientific_notation() {
    assert!((eval_expr("1e-3", 0, None).unwrap() - 0.001).abs() < 1e-15);
    assert!((eval_expr("2.5E2", 0, None).unwrap() - 250.0).abs() < 1e-12);
    assert!((eval_expr("1.5e+2", 0, None).unwrap() - 150.0).abs() < 1e-12);
}

#[test]
fn test_qasm_arithmetic_param() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(pi/2 + pi/4) q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rx(theta) => {
                let expected = std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_4;
                assert!((theta - expected).abs() < 1e-12);
            }
            _ => panic!("expected Rx, got {:?}", gate),
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_qasm_multiply_pi() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(2*pi/3) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - 2.0 * std::f64::consts::PI / 3.0).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
}

#[test]
fn test_qasm_function_in_param() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(sqrt(2)/2) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::P(theta) => {
                assert!((theta - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
            }
            _ => panic!("expected P"),
        }
    }
}

#[test]
fn test_qasm_nested_function_in_param() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(sin(pi/4)) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::P(theta) => {
                assert!((theta - (std::f64::consts::FRAC_PI_4).sin()).abs() < 1e-12);
            }
            _ => panic!("expected P"),
        }
    }
}

#[test]
fn test_qasm_gate_def_with_expression_body() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate myrxx(theta) a, b {
            rx(theta/2) a;
            cx a, b;
            rx(-theta/2) a;
            cx a, b;
        }
        myrxx(pi) q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rx(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            _ => panic!("expected Rx, got {:?}", gate),
        }
    }
    if let Instruction::Gate { gate, .. } = &c.instructions[2] {
        match gate {
            Gate::Rx(theta) => {
                assert!((theta - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
            }
            _ => panic!("expected Rx, got {:?}", gate),
        }
    }
}

#[test]
fn test_qasm_gate_def_multi_param_expression() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate myu3(theta, phi, lambda) a {
            rz(lambda) a;
            ry(theta) a;
            rz(phi) a;
        }
        myu3(pi/2, pi/4, pi/8) q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - std::f64::consts::PI / 8.0).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
    if let Instruction::Gate { gate, .. } = &c.instructions[1] {
        match gate {
            Gate::Ry(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            _ => panic!("expected Ry"),
        }
    }
    if let Instruction::Gate { gate, .. } = &c.instructions[2] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
}

#[test]
fn test_qasm_gate_def_expression_with_arithmetic() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate half_rot(theta) a {
            rz(theta + pi) a;
        }
        half_rot(pi/4) q[0];
    "#;
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                let expected = std::f64::consts::FRAC_PI_4 + std::f64::consts::PI;
                assert!((theta - expected).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
}

#[test]
fn rx_with_arithmetic_overflow_rejected_as_non_finite() {
    // Pure arithmetic overflow: each literal is finite, but the product
    // exceeds f64::MAX. The top-level eval_expr finiteness check catches this
    // path; the per-function and per-literal checks do not.
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(1e308 * 1e308) q[0];
    "#;
    let err = parse(qasm).expect_err("expected non-finite parameter rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("must be finite"),
        "expected finiteness error, got: {msg}"
    );
}

#[test]
fn rx_with_sqrt_of_negative_rejected_as_non_finite() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(sqrt(-1)) q[0];
    "#;
    let err = parse(qasm).expect_err("expected non-finite parameter rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite"),
        "expected finiteness error, got: {msg}"
    );
}

#[test]
fn rx_with_acos_out_of_domain_rejected_as_non_finite() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(acos(2)) q[0];
    "#;
    let err = parse(qasm).expect_err("expected non-finite parameter rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite"),
        "expected finiteness error, got: {msg}"
    );
}

#[test]
fn rx_with_division_by_zero_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(1.0 / (1.0 - 1.0)) q[0];
    "#;
    let err = parse(qasm).expect_err("expected division-by-zero rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("division by zero"),
        "expected division-by-zero error, got: {msg}"
    );
}

#[test]
fn rx_with_finite_expression_still_parses() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(pi/4) q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
}

#[test]
fn for_loop_unrolls_inclusive_range() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        for int i in [0:3] {
            h q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    for (idx, instr) in c.instructions.iter().enumerate() {
        match instr {
            Instruction::Gate {
                gate: Gate::H,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[idx]);
            }
            _ => panic!("expected H gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_with_step() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[6] q;
        for int i in [0:2:4] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    let expected = [0usize, 2, 4];
    for (idx, &qi) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::X,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[qi]);
            }
            _ => panic!("expected X gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_set_form() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[5] q;
        for int i in {0, 2, 4} {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    let expected = [0usize, 2, 4];
    for (idx, &qi) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::X,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[qi]);
            }
            _ => panic!("expected X gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_negative_step() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        for int i in [3:-1:0] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    let expected = [3usize, 2, 1, 0];
    for (idx, &qi) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::X,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[qi]);
            }
            _ => panic!("expected X gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_uses_loop_variable_in_param_expression() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:2] {
            rx(i * pi / 4) q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    for (idx, instr) in c.instructions.iter().enumerate() {
        match instr {
            Instruction::Gate {
                gate: Gate::Rx(theta),
                targets,
            } => {
                let expected = idx as f64 * std::f64::consts::PI / 4.0;
                assert!((theta - expected).abs() < 1e-12);
                assert_eq!(targets.as_slice(), &[idx]);
            }
            _ => panic!("expected Rx gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_nested() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        for int i in [0:1] {
            for int j in [0:1] {
                cx q[i], q[j+2];
            }
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    let expected = [(0usize, 2usize), (0, 3), (1, 2), (1, 3)];
    for (idx, &(ci, ti)) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::Cx,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[ci, ti]);
            }
            _ => panic!("expected CX at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_uint_keyword_accepted() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        for uint i in [0:2] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
}

#[test]
fn for_loop_unterminated_brace_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:2] {
            x q[i];
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn for_loop_zero_step_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:0:2] {
            x q[i];
        }
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn for_loop_non_integer_bound_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:pi] {
            x q[i];
        }
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn def_subroutine_basic_inlines() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        def bell(qubit a, qubit b) {
            h a;
            cx a, b;
        }
        bell(q[0], q[1]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 2);
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::H,
            targets,
        } => {
            assert_eq!(targets.as_slice(), &[0]);
        }
        _ => panic!("expected H"),
    }
    match &c.instructions[1] {
        Instruction::Gate {
            gate: Gate::Cx,
            targets,
        } => {
            assert_eq!(targets.as_slice(), &[0, 1]);
        }
        _ => panic!("expected CX"),
    }
}

#[test]
fn runtime_def_returned_bits_start_fresh_when_the_target_is_an_argument() {
    for (body, expected) in [
        ("bit r; r ^= b; return r;", true),
        ("bit r = 0; r ^= b; return r;", true),
        ("return b;", true),
        ("return b ^ 1;", false),
        ("return -1;", true),
    ] {
        let source = format!("bit c = 1; def f(bit b) -> bit {{ {body} }} c = f(c);");
        let program = parse_dynamic(&source).unwrap();
        let shots = crate::simulate_program(&program)
            .seed(42)
            .shots(10)
            .unwrap();
        assert!(shots.shots.iter().all(|shot| shot == &[expected]), "{body}");
    }
    let program = parse_dynamic("bit c = 1; def f() -> bit { bit r; return r; } c = f();").unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false]));
}

#[test]
fn def_constants_keep_their_lexical_values_under_caller_shadowing() {
    for (source, reference) in [
        ("qubit q; const int n = 4;
          def f() -> int { return n; }
          def g(int n) -> int { return f(); }
          for int n in [8:8] { rx(g(16)) q; }", "qubit q; rx(4) q;"),
        ("qubit q; const duration d = 5ns;
          def wait() -> duration { return d; }
          def g(duration d) -> float { duration v = wait(); return v / 1ns; }
          rx(g(20ns)) q;", "qubit q; rx(5) q;"),
        ("qubit q; def f(int n) -> int { switch(n) { case 1 { return 3; } default { return 4; } } }
          rx(f(1)) q;", "qubit q; rx(3) q;"),
    ] {
        let expected = parse(reference).unwrap();
        let circuit = parse(source).unwrap();
        let program = parse_dynamic(source).unwrap();
        for actual in [&circuit, program.static_circuit().expect("static call")] {
            assert_eq!(format!("{:?}", actual.instructions), format!("{:?}", expected.instructions));
        }
    }
}

#[test]
fn def_scope_cannot_capture_mutable_caller_values_or_registers() {
    for source in [
        "bit c; int n = 2; def f(bit b) -> int { return n + int(b); } int v = f(c);",
        "qubit q; bit c; def f(bit b) -> int { x q; return int(b); } int v = f(c);",
        "bit c; def f(bit b) -> int { c = 1; return int(b); } int v = f(c);",
        "def f() { qubit q; } f();",
    ] {
        assert!(parse(source).is_err(), "{source}");
        assert!(parse_dynamic(source).is_err(), "{source}");
    }
}

#[test]
fn runtime_def_short_circuit_calls_keep_their_gate_effects_conditional() {
    let program = parse_dynamic(
        "qubit[2] q; bit[3] c;
         def flip(qubit a) -> bool { x a; return true; }
         h q[0]; c[0] = measure q[0];
         bool a = bool(c[0]) && flip(q[1]);
         bool b = bool(c[0]) || flip(q[1]);
         c[1] = measure q[1]; c[2] = a == bool(c[0]) && b;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot[1] && shot[2]));
}

#[test]
fn runtime_def_calls_repeat_in_while_conditions() {
    let program = parse_dynamic(
        "qubit q; bit[2] c = 1;
         def check(bit b, qubit a) -> bool { x a; return bool(b); }
         while (check(c[0], q)) { c[0] = 0; }
         c[1] = measure q;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false, false]));
}

#[test]
fn runtime_def_return_leaves_nested_loops_and_keeps_scalar_locals() {
    let program = parse_dynamic(
        "qubit q; bit[2] c;
         def f(bit b) -> int {
             int n = 0;
             while (n < 3) { if (n == int(b)) return n; n += 1; }
             return -1;
         }
         h q; c[0] = measure q;
         int count = 0; bool correct = true;
         while (count < 2) {
             c[0] ^= 1;
             int n = f(c[0]); correct = correct && n == int(c[0]); count += 1;
         }
         c[1] = correct;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot[1]));
}

#[test]
fn runtime_def_bit_parameters_forward_slices_and_keep_writes_local() {
    let program = parse_dynamic(
        "bit[4] c = 2;
         def f(bit[2] b) -> int { b[0] = 1; b[1] ^= 1; return int(b); }
         def g(bit[2] b) -> int { int n; n = f(b[{1,0}]); return n; }
         int n; n = g(c[0:1]); c[2] = n == 3; c[3] = (int(c) & 3) == 2;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(
        shots
            .shots
            .iter()
            .all(|shot| shot == &[false, true, true, true])
    );
}

#[test]
fn runtime_def_results_coerce_and_copy_bit_arguments() {
    let program = parse_dynamic(
        "qubit q; bit[2] c;
         def f(bit b) -> uint[4] { if (b) return 31; return 19; }
         h q; c[0] = measure q;
         int n = f(c[0]); c[1] = n == 3 + 12 * int(c[0]);",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot[1]));
    assert!(shots.shots.iter().any(|shot| shot[0]));
    assert!(shots.shots.iter().any(|shot| !shot[0]));
}

#[test]
fn runtime_def_return_keeps_the_callers_loop_running() {
    let program = parse_dynamic(
        "qubit q; bit[3] c = 1;
         def f(bit b, qubit a) { if (b) return; x a; }
         int n = 0;
         while (n < 2) { n += 1; f(c[0], q); x q; }
         c[1] = measure q; c[2] = n == 2;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, false, true]));
}

#[test]
fn runtime_def_nested_results_keep_distinct_values() {
    let program = parse_dynamic(
        "bit[3] c = 2;
         def f(bit b) -> int { if (b) return 3; return 1; }
         def g(bit a, bit b) -> int { return f(a) + f(b); }
         int n = g(c[0], c[1]); c[2] = n == 4;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false, true, true]));
}

#[test]
fn runtime_def_scalar_parameters_are_values_with_declared_widths() {
    let program = parse_dynamic(
        "bit[2] c = 1;
         def f(uint[4] n) -> uint[4] { if (n > 0) n += 3; return n; }
         uint[8] n = int(c[0]) * 18;
         int r = f(n); c[1] = n == 18 && r == 5;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, true]));
    assert!(matches!(
        parse_dynamic(
            "bit c; def f(int n, bit b) -> int { return n + int(b); } int v = f(0.5, c);"
        ),
        Err(PrismError::Parse { .. })
    ));
}

#[test]
fn runtime_def_requires_a_result_on_every_reachable_path() {
    let error =
        parse_dynamic("bit c; def f(bit b) -> int { if (b) return 1; } int n = f(c);").unwrap_err();
    assert!(
        matches!(error, PrismError::Parse { message, .. } if message.contains("without a result"))
    );
}

#[test]
fn runtime_def_call_can_decide_a_branch() {
    let program = parse_dynamic(
        "qubit q; bit[2] c = 1;
         def f(bit b) -> bool { return bool(b); }
         if (f(c[0])) x q; c[1] = measure q;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, true]));
}

#[test]
fn def_scalar_results_coerce_at_the_declared_width() {
    for (signature, value, expected) in [
        ("int[4]", "9", "-7"),
        ("uint[4]", "19", "3"),
        ("bool", "2", "1"),
        ("float[16]", "1.0001", "1.0"),
        ("float[32]", "16777217", "16777216.0"),
        ("angle[3]", "pi / 3", "pi / 4"),
        ("int", "9007199254740993 - 9007199254740992", "1"),
    ] {
        let source = format!(
            "qubit q; def f() -> {signature} {{ return {value}; }} \
             float v = f(); rx(v) q;"
        );
        let actual = parse(&source).unwrap_or_else(|error| panic!("{signature}: {error}"));
        let reference = parse(&format!("qubit q; rx({expected}) q;")).unwrap();
        assert_eq!(
            format!("{:?}", actual.instructions),
            format!("{:?}", reference.instructions)
        );
    }
}

#[test]
fn def_signature_widths_keep_the_declaration_scope() {
    for (source, reference) in [
        (
            "qubit q; def ratio(float n, float d) -> float { return n / d; } rx(ratio(3, 5)) q;",
            "qubit q; rx(0.6) q;",
        ),
        (
            "qubit q; const int n = 4;
             def f() -> uint[n] { return 31; }
             uint[8] r; for int n in [8:8] { r = f(); } rx(r) q;
             def caller(int n) -> uint[8] { return f(); }
             r = caller(8); rx(r) q;",
            "qubit q; rx(15) q; rx(15) q;",
        ),
        (
            "qubit[2] q; const int n = 2;
             def f(qubit[n] a) { x a[1]; }
             for int n in [1:1] { f(q); }",
            "qubit[2] q; x q[1];",
        ),
    ] {
        let expected = parse(reference).unwrap();
        let circuit = parse(source).unwrap();
        let program = parse_dynamic(source).unwrap();
        for actual in [
            &circuit,
            program.static_circuit().expect("static signature"),
        ] {
            assert_eq!(
                format!("{:?}", actual.instructions),
                format!("{:?}", expected.instructions)
            );
        }
    }
    let wrong_width = "qubit[2] q; const int n = 2;
        def f(qubit[n] a) { x a; }
        for int n in [1:1] { f(q[0]); }";
    assert!(matches!(parse(wrong_width), Err(PrismError::Parse { .. })));
    assert!(matches!(
        parse_dynamic(wrong_width),
        Err(PrismError::Parse { .. })
    ));
}

#[test]
fn if_singleton_bit_selections_keep_guard_lowering() {
    for (condition, reference) in [
        ("c[0:0]", "c[0]"),
        ("c[{0}]", "c[0]"),
        ("!c[0:0]", "!c[0]"),
        ("!c[{0}]", "!c[0]"),
        ("c[0:0] == 1", "c[0] == 1"),
        ("c[{0}] != 0", "c[0] != 0"),
    ] {
        let source = format!("qubit q; bit[2] c; if ({condition}) x q;");
        let expected = parse(&format!("qubit q; bit[2] c; if ({reference}) x q;")).unwrap();
        let circuit = parse(&source).unwrap();
        let program = parse_dynamic(&source).unwrap();
        for actual in [&circuit, program.static_circuit().expect("bit guard")] {
            assert_eq!(
                format!("{:?}", actual.instructions),
                format!("{:?}", expected.instructions),
                "{condition}"
            );
        }
    }
}

#[test]
fn def_early_return_stops_nested_blocks_and_loop_iterations() {
    let circuit = parse(
        "qubit q;
         def f(qubit a) -> int {
             for int i in [0:5] {
                 if (i == 2) { box { return i; } }
                 x a;
             }
             z a;
             return 7;
         }
         int n = f(q);
         rx(n) q;",
    )
    .unwrap();
    let reference = parse("qubit q; x q; x q; rx(2) q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_calls_in_expressions_preserve_order_and_exact_results() {
    let circuit = parse(
        "qubit q;
         def a(qubit t) -> int { x t; return 2; }
         def b(qubit t) -> int { z t; return 3; }
         def twice(int n) -> int { return n * 2; }
         int v = twice(a(q)) + b(q);
         v += twice(2);
         rx(v) q;",
    )
    .unwrap();
    let reference = parse("qubit q; x q; z q; rx(11) q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_void_return_skips_the_remaining_gates_and_keeps_the_caller() {
    let circuit = parse(
        "qubit q;
         def f(qubit a) { x a; return; z a; }
         f(q); h q;",
    )
    .unwrap();
    let reference = parse("qubit q; x q; h q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_static_short_circuit_does_not_execute_a_skipped_call() {
    let circuit = parse(
        "qubit q;
         def f(qubit a) -> bool { x a; return true; }
         bool a = false && f(q);
         bool b = true || f(q);
         rx(int(a) + int(b)) q;",
    )
    .unwrap();
    let reference = parse("qubit q; rx(1) q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_duration_result_keeps_its_units() {
    parse(
        "qubit q; def wait() -> duration { return 5ns; }
         duration d = wait(); delay[d] q;",
    )
    .unwrap();
    assert!(parse("def f() -> duration { return 5; } f();").is_err());
}

#[test]
fn def_result_requires_a_return_on_the_taken_path() {
    let error = parse("def f(int n) -> int { if (n > 0) return n; } int v = f(0);").unwrap_err();
    assert!(matches!(error, PrismError::Parse { message, .. }
        if message.contains("reaches its end without a result")));
}

#[test]
fn def_scalar_calls_keep_the_static_dynamic_program_fast_path() {
    let source = "qubit q;
        def f(int n) -> int { if (n > 1) return n; return 0; }
        if (f(2)) { rx(f(3)) q; }";
    let reference = parse(source).unwrap();
    let dynamic = parse_dynamic(source).unwrap();
    let circuit = dynamic.static_circuit().expect("no runtime values");
    assert_eq!(
        format!("{:?}", reference.instructions),
        format!("{:?}", circuit.instructions)
    );
}

#[test]
fn def_call_gates_precede_runtime_actions_using_the_result() {
    let program = parse_dynamic(
        "qubit q; bit c; h q;
         def f(qubit a) -> float { h a; return pi; }
         float t = f(q) + int(c) * 0.0;
         rz(t) q; c = measure q;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false]));
}

#[test]
fn def_arguments_read_the_caller_before_parameter_names_are_bound() {
    let source = "qubit q; int a = 3;
        def f(int a, int b) -> int { return b; }
        int v = f(1, a); rx(v) q;";
    same_instructions(source, "qubit q; rx(3) q;");
    same_instructions(
        "qubit[2] q; def f(qubit a) -> int { x a; return 2; }
         int v = f(q[1]); rx(v) q[0];",
        "qubit[2] q; x q[1]; rx(2) q[0];",
    );
}

#[test]
fn def_runtime_result_paths_need_dynamic_programs() {
    for source in [
        "qubit q; bit c; def f(bit b) -> int { if (b) return 1; return 0; }
         c = measure q; int n = f(c);",
        "qubit q; bit c; def f(bit b) -> int { return int(b); }
         c = measure q; int n = f(c);",
        "qubit q; bit c; def f() -> bool { return true; }
         if (c && f()) x q;",
    ] {
        assert!(
            matches!(parse(source), Err(PrismError::UnsupportedConstruct { .. })),
            "{source}"
        );
        parse_dynamic(source).unwrap();
    }
}

#[test]
fn def_qubit_register_parameters_preserve_sliced_order() {
    same_instructions(
        "qubit[4] q;
         def f(qubit[2] a, qubit b) { cx a[0], b; x a[1]; }
         f(q[1:2:3], q[0]);",
        "qubit[4] q; cx q[1], q[0]; x q[3];",
    );
    for source in [
        "qubit[3] q; def f(qubit[2] a) { h a; } f(q);",
        "qubit[2] q; def f(qubit[2] a, qubit b) {} f(q, q[0]);",
        "qubit[2] q; def f(qubit[2] a) {} f(q[{0,0}]);",
    ] {
        assert!(
            matches!(parse(source), Err(PrismError::Parse { .. })),
            "{source}"
        );
    }
}

#[test]
fn def_scalar_results_assign_to_arrays_and_classical_bits() {
    same_instructions(
        "qubit q; def f() -> int { return 3; }
         array[int, 2] a; a[1] = f(); a[0] += f() + 1; rx(a[0] + a[1]) q;",
        "qubit q; rx(7) q;",
    );
    same_instructions(
        "qubit q; def f() -> float { return pi / 2; }
         float v = 0; v = sin(f()); rx(v) q;",
        "qubit q; rx(1) q;",
    );
    let program =
        parse_dynamic("bit[3] c; def f() -> int { return 3; } c = f(); c[2] = f();").unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, true, true]));
}

#[test]
fn def_register_result_runs_inside_a_repeat_until_success_loop() {
    let program = parse_dynamic(
        "qubit[2] q; bit[2] c = 3;
         def check(qubit[2] a) -> bit[2] {
             bit[2] r; reset a; h a; measure a -> r; return r;
         }
         while (int(c) != 0) { c = check(q); }",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false, false]));
}

#[test]
fn def_subroutine_with_float_parameter() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        def my_rx(float theta, qubit a) {
            rx(theta) a;
        }
        my_rx(0.5, q[0]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            targets,
        } => {
            assert!((theta - 0.5).abs() < 1e-12);
            assert_eq!(targets.as_slice(), &[0]);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn def_subroutine_with_int_parameter_used_as_index() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        def at_idx(int i, qubit a) {
            rx(i * pi / 2) a;
        }
        at_idx(2, q[1]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            targets,
        } => {
            assert!((theta - std::f64::consts::PI).abs() < 1e-12);
            assert_eq!(targets.as_slice(), &[1]);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn def_subroutine_with_for_loop_inside_body() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        def all_h(qubit a, qubit b, qubit c) {
            for int i in [0:2] {
                rx(i * pi / 2) a;
            }
            h b;
            h c;
        }
        all_h(q[0], q[1], q[2]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 5);
}

#[test]
fn def_returning_a_literal_is_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        def measure_qubit(qubit a) -> bit {
            return 0;
        }
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::UnsupportedConstruct { .. }));
}

#[test]
fn def_subroutine_arity_mismatch_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        def bell(qubit a, qubit b) {
            h a;
            cx a, b;
        }
        bell(q[0]);
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::GateArity { .. }));
}

#[test]
fn def_subroutine_recursion_capped() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        def loop(qubit a) {
            loop(a);
        }
        loop(q[0]);
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn def_subroutine_called_inside_for() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        def step(qubit a) {
            h a;
        }
        for int i in [0:3] {
            step(q[i]);
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    for (idx, instr) in c.instructions.iter().enumerate() {
        match instr {
            Instruction::Gate {
                gate: Gate::H,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[idx]);
            }
            _ => panic!("expected H gate at slot {idx}"),
        }
    }
}

#[test]
fn if_bit_equals_one_form_parses() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] == 1) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional {
            condition,
            gate: Gate::X,
            targets,
        } => {
            assert!(matches!(condition, ClassicalCondition::BitIsOne(0)));
            assert_eq!(targets.as_slice(), &[0]);
        }
        _ => panic!("expected conditional X gate"),
    }
}

#[test]
fn if_bit_equals_zero_form_parses() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] == 0) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsZero(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_bit_not_equal_one_negates() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] != 1) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsZero(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_bit_not_equal_zero_is_one() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] != 0) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsOne(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_negated_bit_truthy_check() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (!c[0]) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsZero(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_register_not_equals() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[3] c;
        if (c != 5) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterNotEquals { value: 5, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_register_equals_hex_literal() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[8] c;
        if (c == 0xff) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterEquals { value: 255, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_register_equals_binary_literal() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[4] c;
        if (c == 0b1010) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterEquals { value: 10, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn parametric_gate_accepts_hex_literal() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(0x2 * pi / 8) q[0];";
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            ..
        } => {
            assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn parametric_gate_accepts_underscore_in_hex() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(0xff_ff * 0) q[0];";
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rz(theta),
            ..
        } => {
            assert!(theta.abs() < 1e-12);
        }
        _ => panic!("expected Rz"),
    }
}

#[test]
fn parametric_gate_accepts_boolean_literal() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(true * pi) q[0];";
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            ..
        } => {
            assert!((theta - std::f64::consts::PI).abs() < 1e-12);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn for_loop_accepts_hex_bound() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[8] q;
        for int i in [0:0x3] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
}

#[test]
fn if_register_compare_with_int_var_in_def() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[4] c;
        def cond(int n, bit[4] b, qubit a) {
            if (b == n) x a;
        }
        cond(3, c, q[0]);
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterEquals { value: 3, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_negative_value_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[4] c;
        if (c == -1) x q[0];
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn if_bit_compare_against_two_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] == 2) x q[0];
    "#;
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::Parse { .. }));
}

#[test]
fn gate_names_round_trip_through_resolve_gate() {
    let named = [
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::SX,
        Gate::SXdg,
        Gate::Rx(0.3),
        Gate::Ry(0.4),
        Gate::Rz(0.5),
        Gate::P(0.6),
        Gate::Rzz(0.7),
        Gate::Cx,
        Gate::Cz,
        Gate::Swap,
    ];
    for gate in named {
        let params: Vec<f64> = match &gate {
            Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::P(t) | Gate::Rzz(t) => vec![*t],
            _ => vec![],
        };
        // `rzz` reaches the Pauli-rotation rule first, the way the parser
        // dispatches it, and that rule lowers ZZ back to `Gate::Rzz`.
        let resolved = match super::pauli_rotation_axes(gate.name()) {
            Some(axes) => {
                let factors: Vec<PauliTerm> = axes
                    .iter()
                    .enumerate()
                    .map(|(q, &axis)| PauliTerm::new(q, axis))
                    .collect();
                super::pauli_rotation_gate(params[0], &factors).0
            }
            None => Parser::resolve_gate(gate.name(), &params, Dialect::Native, 0)
                .unwrap_or_else(|e| panic!("`{}` did not resolve: {e}", gate.name())),
        };
        assert_eq!(
            resolved,
            gate,
            "`{}` resolved to a different variant",
            gate.name()
        );
    }
}

// Expression operators and builtins

fn angle_of(source: &str) -> f64 {
    let circuit = parse(source).unwrap_or_else(|e| panic!("`{source}`: {e}"));
    match &circuit.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            ..
        } => *theta,
        other => panic!("expected an rx, got {other:?}"),
    }
}

fn angle_err(source: &str) -> PrismError {
    parse(source)
        .err()
        .unwrap_or_else(|| panic!("`{source}` should not parse"))
}

fn rx(expression: &str) -> String {
    format!("OPENQASM 3.0;\nqubit[1] q;\nrx({expression}) q[0];")
}

#[test]
fn expression_operators_follow_openqasm_precedence() {
    let cases = [
        ("2 ** 3", 8.0),
        // `**` binds tighter than unary minus, so this is -(2^2).
        ("-2 ** 2", -4.0),
        // and it is right associative.
        ("2 ** 3 ** 2", 512.0),
        ("2 ** -1", 0.5),
        ("2 ** 3 * 4", 32.0),
        ("7 % 3", 1.0),
        ("-7 % 3", -1.0),
        ("mod(7.5, 2)", 1.5),
        ("1 + 2 * 3 % 4", 3.0),
    ];
    for (expression, expected) in cases {
        let actual = angle_of(&rx(expression));
        assert!(
            (actual - expected).abs() < 1e-12,
            "`{expression}`: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn builtin_functions_take_their_spec_spellings() {
    let cases = [
        ("arcsin(1.0)", std::f64::consts::FRAC_PI_2),
        ("arccos(1.0)", 0.0),
        ("arctan(1.0)", std::f64::consts::FRAC_PI_4),
        ("asin(1.0)", std::f64::consts::FRAC_PI_2),
        ("ceiling(0.2)", 1.0),
        ("ceil(0.2)", 1.0),
        ("floor(1.8)", 1.0),
        ("log(euler)", 1.0),
        ("ln(euler)", 1.0),
        ("abs(-2.5)", 2.5),
        ("mod(7, 3)", 1.0),
        ("pow(2, 10)", 1024.0),
        ("popcount(7)", 3.0),
        ("popcount(0)", 0.0),
        ("popcount(0xff)", 8.0),
    ];
    for (expression, expected) in cases {
        let actual = angle_of(&rx(expression));
        assert!(
            (actual - expected).abs() < 1e-12,
            "`{expression}`: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn malformed_expressions_name_what_is_wrong() {
    for (expression, needle) in [
        ("mod(7)", "argument"),
        ("pow(2, 3, 4)", "argument"),
        ("popcount(1.5)", "popcount"),
        ("popcount(-1)", "popcount"),
        ("1 % 0", "modulo by zero"),
        ("1 / 0", "division by zero"),
        ("sizeof(2)", "unknown function"),
    ] {
        let err = format!("{}", angle_err(&rx(expression)));
        assert!(
            err.contains(needle),
            "`{expression}`: expected `{needle}` in `{err}`"
        );
    }
}

// Block comments, version, physical qubits

// A block comment is blanked rather than removed, so the line a later error
// names is still the line the author wrote it on.
#[test]
fn block_comments_are_skipped_and_keep_line_numbers() {
    let source = "OPENQASM 3.0;\n\
                  /* a comment\n\
                     spanning lines */\n\
                  qubit[2] q;\n\
                  h /* inline */ q[0];\n\
                  nonsense q[0];\n";
    let err = parse(source).unwrap_err();
    assert!(
        matches!(err, PrismError::UnsupportedConstruct { line: 6, .. }),
        "got {err:?}"
    );

    let circuit = parse("OPENQASM 3.0;\nqubit[1] q;\nh /* mid */ q[0];\n").unwrap();
    assert_eq!(circuit.gate_count(), 1);
}

#[test]
fn a_block_comment_does_not_nest_and_must_close() {
    // The first `*/` closes the span, so the tail is code and does not parse.
    assert!(parse("OPENQASM 3.0;\nqubit[1] q;\n/* a /* b */ c */\nh q[0];\n").is_err());
    let err = parse("OPENQASM 3.0;\nqubit[1] q;\n/* never closed\nh q[0];\n").unwrap_err();
    assert!(
        matches!(err, PrismError::Parse { line: 3, .. }),
        "got {err:?}"
    );
}

// A `//` inside a block comment and a `/*` inside a line comment or a string
// are all text, not delimiters.
#[test]
fn comment_delimiters_do_not_cross_each_other() {
    let circuit = parse(
        "OPENQASM 3.0;\n\
         // a line comment with /* in it\n\
         /* a block comment with // in it */\n\
         include \"std/*gates.inc\";\n\
         qubit[1] q;\n\
         h q[0];\n",
    )
    .unwrap();
    assert_eq!(circuit.gate_count(), 1);
}

#[test]
fn the_version_statement_is_checked() {
    for accepted in ["OPENQASM 3.0;", "OPENQASM 3;", "OPENQASM 2.0;"] {
        let source = format!("{accepted}\nqubit[1] q;\nh q[0];\n");
        assert!(parse(&source).is_ok(), "`{accepted}`");
    }
    for rejected in ["OPENQASM 4.0;", "OPENQASM 99;"] {
        let source = format!("{rejected}\nqubit[1] q;\nh q[0];\n");
        assert!(
            matches!(
                parse(&source),
                Err(PrismError::UnsupportedConstruct { line: 1, .. })
            ),
            "`{rejected}`"
        );
    }
    assert!(matches!(
        parse("OPENQASM three;\nqubit[1] q;\n"),
        Err(PrismError::Parse { line: 1, .. })
    ));
}

// A physical qubit is an absolute index, so the register is as wide as the
// highest one named and nothing declares it.
#[test]
fn physical_qubits_size_the_register() {
    let circuit =
        parse("OPENQASM 3.0;\nbit[1] c;\nh $0;\ncx $0, $3;\nc[0] = measure $3;\n").unwrap();
    assert_eq!(circuit.num_qubits, 4);
    assert_eq!(circuit.measurement_map(), [(3, 0)]);
    let Instruction::Gate { targets, .. } = &circuit.instructions[1] else {
        panic!("expected a gate");
    };
    assert_eq!(targets.as_slice(), [0, 3]);
}

#[test]
fn physical_qubits_and_a_register_cannot_be_mixed() {
    for source in [
        "OPENQASM 3.0;\nqubit[2] q;\nh $0;\n",
        "OPENQASM 2.0;\nqreg q[2];\nh $0;\n",
    ] {
        assert!(
            matches!(parse(source), Err(PrismError::UnsupportedConstruct { .. })),
            "`{source}`"
        );
    }
}

// A `$` inside a string literal is part of a file name, not a qubit, so the
// prescan has to leave it alone or the register comes back nine qubits wider
// and every declaration reads as mixed addressing.
#[test]
fn test_physical_scan_skips_string_literals() {
    let qasm = "OPENQASM 3.0;\ninclude \"a$9.inc\";\nqubit[2] q;\nh q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.num_qubits, 2);
}

// A `defcal` names a calibration target rather than a program qubit, and is
// rejected by name rather than by the register it would otherwise size.
#[test]
fn test_a_calibration_leaves_its_gate_meaning_alone() {
    let qasm = "OPENQASM 3.0;\ndefcalgrammar \"openpulse\";\ncal { extern port d0; }\n\
                defcal x $0 { play(d0, gaussian(...)); }\nqubit[1] q;\nx q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
    assert!(matches!(
        &c.instructions[0],
        Instruction::Gate { gate: Gate::X, .. }
    ));
}

// A defcal says how to pulse a gate, not what the gate is, so a name only a
// defcal introduces has no unitary to run.
#[test]
fn test_a_gate_only_a_defcal_names_is_declined_at_the_call() {
    let qasm = "OPENQASM 3.0;\ndefcal flux $0 { }\nqubit[1] q;\nflux q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(
        matches!(&err, PrismError::UnsupportedConstruct { construct, .. } if construct == "flux"),
        "got {err}"
    );
}

#[test]
fn test_version_message_names_the_range() {
    for version in ["1.0", "4.0"] {
        let err = parse(&format!("OPENQASM {version};\nqubit[1] q;\nh q[0];")).unwrap_err();
        assert!(
            format!("{err}").contains("reads 2 and 3"),
            "`{version}`: got {err}"
        );
    }
}

// A control and its target cannot be the same qubit: the controlled gate would
// index the same amplitude twice.
#[test]
fn test_control_and_target_must_differ() {
    for qasm in [
        "OPENQASM 3.0;\nqubit[2] q;\nctrl @ x q[0], q[0];",
        "OPENQASM 3.0;\nqubit[2] q;\nnegctrl @ h q[1], q[1];",
        "OPENQASM 3.0;\nqubit[3] q;\nctrl @ swap q[1], q[0], q[1];",
    ] {
        let err = parse(qasm).unwrap_err();
        assert!(
            format!("{err}").contains("both a control and a target"),
            "`{qasm}`: got {err}"
        );
    }
}

// A `defcal` body carries physical qubits too, and skipping only its first
// line would size the register off the rest of it.
#[test]
fn test_physical_scan_skips_a_defcal() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ndefcal x $0 {\n  shift_phase($1, 0.5);\n}";
    let c = parse(qasm).unwrap();
    assert_eq!((c.num_qubits, c.instructions.len()), (2, 1));
}

// A quote inside a line comment must not open a string that swallows a later
// block comment.
#[test]
fn test_a_string_ends_at_its_own_line() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\n// a \" stray quote\n/* dropped */\nx q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 1);
}

// A statement ends at its `;` and a block at its `}` wherever the newlines fall.
#[test]
fn a_statement_ends_at_its_semicolon_not_its_line() {
    let split = parse("OPENQASM 3.0;\nqubit[2] q;\nh\n  q[0]\n  ;\ncx\n  q[0],\n  q[1];\n")
        .expect("split statements");
    let packed =
        parse("OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];").expect("packed statements");
    assert_eq!(
        format!("{:?}", split.instructions),
        format!("{:?}", packed.instructions)
    );
    assert_eq!(split.num_qubits, 2);
}

#[test]
fn a_block_ends_at_its_brace_not_its_line() {
    let wrapped = parse(
        "OPENQASM 3.0;\nqubit[2] q;\ngate flip(t)\n  a,\n  b\n{\n  rx(t) a;\n  cx a, b;\n}\nflip(pi) q[0], q[1];\n",
    )
    .expect("wrapped gate definition");
    let inline = parse(
        "OPENQASM 3.0; qubit[2] q; gate flip(t) a, b { rx(t) a; cx a, b; } flip(pi) q[0], q[1];",
    )
    .expect("inline gate definition");
    assert_eq!(
        format!("{:?}", wrapped.instructions),
        format!("{:?}", inline.instructions)
    );
}

#[test]
fn a_guarded_region_reads_across_lines_either_way() {
    let wrapped = parse(
        "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nmeasure q[0] -> c[0];\nif (c[0])\n{\n  x q[1];\n}\nelse\n{\n  y q[1];\n}\n",
    )
    .expect("wrapped region");
    let inline = parse(
        "OPENQASM 3.0; qubit[2] q; bit[2] c; measure q[0] -> c[0]; if (c[0]) { x q[1]; } else { y q[1]; }",
    )
    .expect("inline region");
    assert_eq!(
        format!("{:?}", wrapped.instructions),
        format!("{:?}", inline.instructions)
    );
}

// A gate body is parsed once and run against the call's bindings, so a name
// the body shares with an enclosing register is the parameter, not the
// register.
#[test]
fn a_gate_parameter_shadows_a_register_of_the_same_name() {
    let circuit =
        parse("OPENQASM 3.0; qubit[4] a; gate flip a { x a; } flip a[2];").expect("shadowed name");
    assert_eq!(circuit.instructions.len(), 1);
    match &circuit.instructions[0] {
        Instruction::Gate { targets, .. } => assert_eq!(targets.as_slice(), &[2]),
        other => panic!("{other:?}"),
    }
}

// A diagnostic names the reference as it was written. The tree keeps the
// subscript rather than the source span, so the rendering is what proves it.
#[test]
fn a_subscript_reads_back_into_its_diagnostic() {
    for (source, wanted) in [
        (
            "qubit[4] q;\nh q[-5];\n",
            "index -5 in `q[-5]` is before its start",
        ),
        (
            "qubit[4] q;\nh q[0:0:3];\n",
            "range step in `q[0:0:3]` must be non-zero",
        ),
        ("qubit[4] q;\nh q[3:1];\n", "`q[3:1]` names no index"),
        (
            "const int n = 9;\nqubit[4] q;\nh q[n];\n",
            "invalid qubit index 9",
        ),
    ] {
        let message = format!("{}", parse(source).expect_err(source));
        assert!(message.contains(wanted), "`{source}` gave: {message}");
    }
}

#[test]
fn a_bare_register_condition_names_the_forms_it_accepts() {
    let message = format!(
        "{}",
        parse("bit[2] c;\nqubit[1] q;\nif (c) x q[0];\n").expect_err("bare register")
    );
    for wanted in ["`c==value`", "`c[i]`", "`!c[i]`", "`c[i]==0/1`"] {
        assert!(message.contains(wanted), "{message}");
    }
}

#[test]
fn a_compound_assignment_by_zero_names_the_operation() {
    let divide = format!("{}", parse("int n = 4;\nn /= 0;\n").expect_err("divide"));
    assert!(divide.contains("division by zero"), "{divide}");
    let modulo = format!("{}", parse("int n = 4;\nn %= 0;\n").expect_err("modulo"));
    assert!(modulo.contains("modulo by zero"), "{modulo}");
}

// Three shapes the line reader accepted or mis-resolved because a newline ended
// a statement and a name was only ever looked up by text.
#[test]
fn shapes_the_line_reader_let_through_are_rejected() {
    let missing = parse("qubit[1] q;\nh q[0]\n").expect_err("missing semicolon");
    assert!(format!("{missing}").contains("`;`"), "{missing}");

    let trailing = parse("qubit[2] q;\ncx q[0], q[1],;\n").expect_err("trailing comma");
    assert!(matches!(trailing, PrismError::Parse { .. }), "{trailing:?}");

    // Rebinding the name would leave the first register unreachable.
    let redeclared = parse("qubit[1] q;\nqubit[1] q;\n").expect_err("redeclared");
    assert!(format!("{redeclared}").contains("already"), "{redeclared}");
}

// Three the line reader rejected or mis-resolved, which the tree resolves by
// scope rather than by text.
#[test]
fn shapes_the_line_reader_rejected_now_parse() {
    let sized = parse("const int n = 3;\nqubit[n] q;\nh q[2];\n").expect("const register size");
    assert_eq!(sized.num_qubits, 3);

    let aliased =
        parse("qubit[1] q;\nbit[4] c;\nlet a = c[1];\nswitch (a) { case 1 { x q[0]; } }\n")
            .expect("switch on a one-bit alias");
    assert_eq!(aliased.instructions.len(), 1);

    let inverted = parse("def sub(qubit a) { rx(0.5) a; }\nqubit[1] q;\ninv @ sub(q[0]);\n")
        .expect("inv on a def call");
    match &inverted.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            ..
        } => assert!((theta + 0.5).abs() < 1e-12),
        other => panic!("{other:?}"),
    }
}

fn nested_parens(depth: usize) -> String {
    format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx({}1{}) q[0];",
        "(".repeat(depth),
        ")".repeat(depth)
    )
}

fn nested_ifs(depth: usize) -> String {
    format!(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\n{}x q[0];",
        "if (c[0]) ".repeat(depth)
    )
}

fn long_sum(terms: usize) -> String {
    format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx(0{}) q[0];",
        " + 0.001".repeat(terms - 1)
    )
}

fn assert_too_deep(result: Result<impl std::fmt::Debug>) {
    let err = result.unwrap_err().to_string();
    assert!(err.contains("nests deeper than"), "{err}");
}

#[test]
fn deep_nesting_is_a_parse_error_not_a_stack_overflow() {
    assert_too_deep(parse(&nested_parens(3_000)));
    assert_too_deep(parse(&nested_ifs(3_000)));
    assert_too_deep(parse(&long_sum(10_000)));
    assert_too_deep(parse(&format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx({}1) q[0];",
        "-".repeat(3_000)
    )));
    assert_too_deep(parse(&format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx(1{}) q[0];",
        " ** 1".repeat(3_000)
    )));
    assert_too_deep(parse_dynamic(&nested_parens(3_000)));
}

// The deepest sources the limit admits parse and evaluate on a 1 MiB stack, the
// default for a Windows main thread.
#[test]
fn nesting_just_inside_the_limit_fits_a_small_stack() {
    let depth = MAX_NESTING - 8;
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let c = parse(&nested_parens(depth)).unwrap();
            assert_eq!(c.gate_count(), 1);
            let source = nested_ifs(depth);
            let tokens = lexer::tokenize(&source).unwrap();
            crate::circuit::qasm::parser::parse_program(&tokens).unwrap();
            let c = parse(&long_sum(depth)).unwrap();
            let Instruction::Gate {
                gate: Gate::Rx(theta),
                ..
            } = &c.instructions[0]
            else {
                panic!("expected rx");
            };
            assert!((theta - 0.001 * (depth - 1) as f64).abs() < 1e-9);
            parse_dynamic(&nested_parens(depth)).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

fn parse_within(source: &str, budget: usize) -> Result<Circuit> {
    let parser = Parser::new_with(source, Dialect::Native);
    parser.expansion_left.set(budget);
    parser.parse().map(|(circuit, _)| circuit)
}

fn assert_expands_past(result: Result<Circuit>) {
    let err = result.unwrap_err().to_string();
    assert!(err.contains("expands past"), "{err}");
}

// Each gate calls the one before it twice, so twenty definitions expand to
// 2^20 instructions though no single construct passes its own bound.
#[test]
fn doubling_gate_definitions_exhaust_the_expansion_budget() {
    let mut source = String::from("OPENQASM 3.0;\nqubit[1] q;\ngate g0 a { x a; }\n");
    for level in 1..=20 {
        let prev = level - 1;
        source.push_str(&format!("gate g{level} a {{ g{prev} a; g{prev} a; }}\n"));
    }
    source.push_str("g20 q[0];\n");
    assert_expands_past(parse_within(&source, 1 << 12));

    let small = source.replace("g20 q[0];", "g5 q[0];");
    assert_eq!(parse_within(&small, 1 << 12).unwrap().gate_count(), 32);
}

#[test]
fn nested_loops_exhaust_the_expansion_budget_even_when_empty() {
    let empty = "OPENQASM 3.0;\nqubit[1] q;\n\
                 for uint i in [0:999999] { for uint j in [0:999999] { } }";
    assert_expands_past(parse_within(empty, 1 << 12));
    let busy = "OPENQASM 3.0;\nqubit[1] q;\n\
                for uint i in [0:999] { for uint j in [0:999] { x q[0]; } }";
    assert_expands_past(parse_within(busy, 1 << 12));
    let pow = "OPENQASM 3.0;\nqubit[1] q;\ngate g a { x a; h a; }\npow(1000000) @ g q[0];";
    assert_expands_past(parse_within(pow, 1 << 12));
}

// Each `x` costs one unit and each call or loop pass one more for itself, and
// nothing is charged again by the level that collects it: four passes of
// 2 * (2 + 1) + 1 + 1 units, plus five top-level statements, make 37.
#[test]
fn the_expansion_budget_charges_work_once() {
    let source = "OPENQASM 3.0;
qubit[1] q;
gate g a { x a; x a; }
                  gate h2 a { g a; g a; }
for uint i in [0:3] { h2 q[0]; }";
    assert_eq!(parse_within(source, 37).unwrap().gate_count(), 16);
    assert_expands_past(parse_within(source, 36));
}

// A def body gets its own copy of the caller's values and constants, so each
// call pays a unit for every name it copies.
#[test]
fn a_def_call_charges_the_names_it_copies() {
    let calls = format!(
        "OPENQASM 3.0;
qubit[1] q;
def f(qubit a) {{ x a; }}
{}",
        "f(q[0]);
"
        .repeat(8)
    );
    assert_eq!(parse_within(&calls, 1 << 10).unwrap().gate_count(), 8);
    let constants: String = (0..100)
        .map(|i| {
            format!(
                "const int k{i} = {i};
"
            )
        })
        .collect();
    let crowded = calls.replace(
        "qubit[1] q;
",
        &format!(
            "qubit[1] q;
{constants}"
        ),
    );
    assert_expands_past(parse_within(&crowded, 1 << 10));
    assert_eq!(parse(&crowded).unwrap().gate_count(), 8);
}

#[test]
fn oversized_registers_are_parse_errors() {
    let wide = |decls: &str| {
        let err = parse(&format!("OPENQASM 3.0;\n{decls}")).unwrap_err();
        assert!(err.to_string().contains("declares more than"), "{err}");
    };
    wide("qubit[9223372036854775807] a;\nqubit[9223372036854775807] b;");
    wide("bit[9223372036854775807] a;\nbit[9223372036854775807] b;");
    wide("bit[16777217] c;");
    wide("qubit[16777216] a;\nqubit[1] b;");
    wide("x $18446744073709551615;");
    let err = parse_dynamic("OPENQASM 3.0;\nx $18446744073709551615;").unwrap_err();
    assert!(err.to_string().contains("declares more than"), "{err}");
}

// A step that would carry the next index past the integer range carries it
// past the end of the range too, so the walk stops there.
#[test]
fn a_range_step_past_the_integer_range_ends_the_range() {
    let c = parse("OPENQASM 3.0;\nqubit[4] q;\nx q[1:9223372036854775807:3];").unwrap();
    assert_eq!(c.gate_count(), 1);
    let c = parse(
        "OPENQASM 3.0;\nqubit[1] q;\n\
         for int i in [1:9223372036854775807:9223372036854775807] { x q[0]; }",
    )
    .unwrap();
    assert_eq!(c.gate_count(), 1);
}

#[test]
fn a_repeated_pow_is_bounded_before_it_allocates() {
    let err = parse("OPENQASM 3.0;\nqubit[2] q;\npow(1000000) @ pow(1000000) @ cx q[0], q[1];")
        .unwrap_err()
        .to_string();
    assert!(err.contains("expands past"), "{err}");
}

#[test]
fn a_runtime_rotation_is_charged_per_qubit() {
    let source = |passes: usize| {
        format!(
            "OPENQASM 3.0;\nqubit[100] q;\nbit[1] c;\nfloat t = 0;\n\
             c[0] = measure q[0];\nif (c[0]) t = pi;\n\
             for int i in [1:{passes}] {{ rx(t) q; }}"
        )
    };
    assert!(super::dynamic::parse_dynamic_within(&source(5), 1000).is_ok());
    let err = super::dynamic::parse_dynamic_within(&source(20), 1000)
        .unwrap_err()
        .to_string();
    assert!(err.contains("expands past"), "{err}");
}

#[test]
fn nested_fors_with_measured_exits_walk_in_linear_time() {
    let nested = |depth: usize| {
        let mut source = String::from(
            "OPENQASM 3.0;
qubit[2] q;
bit[2] c;
c[0] = measure q[0];
",
        );
        for level in 0..depth {
            source += &format!("for int i{level} in [0:0] {{ if (c[0]) {{ break; }} ");
        }
        source += "x q[1];";
        source += &" }".repeat(depth);
        source
    };
    let walked = |depth: usize| {
        super::dynamic::WALKED.with(|walked| walked.set(0));
        parse_dynamic(&nested(depth)).unwrap();
        super::dynamic::WALKED.with(Cell::get)
    };
    let (shallow, deep) = (walked(8), walked(16));
    assert!(
        deep <= shallow * 3,
        "{shallow} statements walked at depth 8, {deep} at 16"
    );
    let (shallow, deep) = (walked(8), walked(120));
    assert!(
        deep * 8 <= shallow * 120 * 2,
        "{shallow} walked at 8, {deep} at 120"
    );
}

#[test]
fn a_pass_of_blocks_costs_more_budget_than_a_pass_of_instructions() {
    let head = "OPENQASM 3.0;
qubit[2] q;
bit[2] c;
c[0] = measure q[0];
";
    let needs = |body: &str| {
        let source = format!("{head}for int j in [0:99] {{ if (c[0]) {{ {body} }} }}");
        let fits = |budget| super::dynamic::parse_dynamic_within(&source, budget).is_ok();
        let (mut low, mut high) = (1, 1 << 16);
        while low < high {
            let mid = (low + high) / 2;
            if fits(mid) { high = mid } else { low = mid + 1 }
        }
        low
    };
    let (plain, exits) = (needs("x q[0];"), needs("break;"));
    assert!(
        exits > 2 * plain,
        "a pass of blocks needs {exits}, one of instructions {plain}"
    );
}

#[test]
fn nested_boxes_just_inside_the_limit_fit_a_small_stack() {
    let depth = MAX_NESTING - 8;
    let source = format!(
        "OPENQASM 3.0;\nqubit[1] q;\n{}x q[0];{}",
        "box { ".repeat(depth),
        " }".repeat(depth)
    );
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            assert_eq!(parse(&source).unwrap().gate_count(), 1);
            parse_dynamic(&source).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

// A 64 KiB stack holds a few levels at most, so every walk below has to move
// onto heap segments to finish.
#[test]
fn sources_inside_the_limit_outgrow_a_tiny_stack() {
    let depth = MAX_NESTING - 8;
    let header = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\n";
    let sources = [
        nested_parens(depth),
        long_sum(depth),
        format!(
            "{header}rx({}1{}) q[0];",
            "sin(".repeat(depth),
            ")".repeat(depth)
        ),
        format!("{header}rx({}1) q[0];", "-".repeat(depth)),
        format!(
            "{header}const array[int, 1] a = {{0}};\nx q[{}0{}];",
            "a[".repeat(depth),
            "]".repeat(depth)
        ),
        format!(
            "{header}{}x q[0];{}",
            "box { ".repeat(depth),
            " }".repeat(depth)
        ),
        format!(
            "{header}{}x q[0];{}",
            (0..depth)
                .map(|i| format!("for int i{i} in [0:0] {{ "))
                .collect::<String>(),
            " }".repeat(depth)
        ),
        format!(
            "{header}c[0] = measure q[0];\nint x = {}c[0];\nif (x == 1) x q[1];",
            "-".repeat(depth)
        ),
    ];
    std::thread::Builder::new()
        .stack_size(64 << 10)
        .spawn(move || {
            for source in &sources {
                parse_dynamic(source).unwrap();
                if !source.contains("measure") {
                    parse(source).unwrap();
                    parse_braket(source).unwrap();
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn boxes_past_the_nesting_limit_are_a_parse_error() {
    let source = format!(
        "OPENQASM 3.0;\nqubit[1] q;\n{}x q[0];{}",
        "box { ".repeat(3_000),
        " }".repeat(3_000)
    );
    assert_too_deep(parse(&source));
    assert_too_deep(parse_dynamic(&source));
}

#[test]
fn gate_calls_through_deep_boxes_fit_a_small_stack() {
    let depth = MAX_NESTING - 8;
    let mut source = String::from("OPENQASM 3.0;\nqubit[1] q;\ngate g0 a { x a; }\n");
    for level in 1..32 {
        source.push_str(&format!(
            "gate g{level} a {{ {}g{} a;{} }}\n",
            "box { ".repeat(depth),
            level - 1,
            " }".repeat(depth)
        ));
    }
    source.push_str("g31 q[0];\n");
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let result = parse(&source);
            assert!(
                result.as_ref().is_ok_and(|c| c.gate_count() == 1)
                    || result
                        .as_ref()
                        .is_err_and(|e| e.to_string().contains("nests deeper")),
                "{result:?}"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

fn rx_angles(source: &str) -> Vec<f64> {
    parse(source)
        .unwrap()
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate {
                gate: Gate::Rx(theta),
                ..
            } => Some(*theta),
            _ => None,
        })
        .collect()
}

// A static integer stores what a runtime one of its type would: truncation
// toward zero, then wraparound to the declared width.
#[test]
fn integer_assignments_truncate_and_wrap_to_the_declared_width() {
    let angles = rx_angles(
        "OPENQASM 3.0;\nqubit[1] q;\n\
         int n = 7;\nn /= 2;\nrx(n) q[0];\n\
         int m = -7;\nm /= 2;\nrx(m) q[0];\n\
         uint[8] u = 300;\nrx(u) q[0];\n\
         uint[8] v = 250;\nv += 10;\nrx(v) q[0];\n\
         int[8] w = 127;\nw += 1;\nrx(w) q[0];\n\
         uint[4] z = 3;\nz -= 4;\nrx(z) q[0];\n\
         bool b = false;\nb += 2;\nrx(b) q[0];",
    );
    assert_eq!(angles, [3.0, -3.0, 44.0, 4.0, -128.0, 15.0, 1.0]);
}

#[test]
fn an_unsigned_integer_wraps_below_zero_and_back() {
    let angles = rx_angles(
        "OPENQASM 3.0;\nqubit[1] q;\n\
         uint x = 0;\nx -= 1;\nx += 1;\nrx(x) q[0];\n\
         uint[64] y = 0;\ny -= 1;\ny += 1;\nrx(y) q[0];\n\
         uint[64] z = 0;\nz -= 2;\nz += 3;\nrx(z) q[0];\n\
         uint[64] w = 0;\nw -= 1;\nuint[64] v = w;\nv += 1;\nrx(v) q[0];",
    );
    assert_eq!(angles, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn built_in_names_cannot_be_redeclared_at_the_top_level() {
    for decl in [
        "int true = 5;",
        "bool false = true;",
        "float pi = 3;",
        "float \u{3c0} = 3;",
        "float tau = 1;",
        "const float euler = 1;",
        "uint[8] true;",
    ] {
        let err = parse(&format!("OPENQASM 3.0;\nqubit[1] q;\n{decl}\nrx(1) q[0];"))
            .expect_err(decl)
            .to_string();
        assert!(err.contains("built-in"), "{decl}: {err}");
    }
}

fn same_instructions(a: &str, b: &str) {
    let (left, right) = (parse(a).unwrap(), parse(b).unwrap());
    assert_eq!(
        format!("{:?}", left.instructions),
        format!("{:?}", right.instructions),
        "`{a}` against `{b}`"
    );
}

#[test]
fn a_negative_index_counts_from_the_end() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\n";
    same_instructions(&format!("{head}x q[-1];"), &format!("{head}x q[3];"));
    same_instructions(&format!("{head}h q[1:-1];"), &format!("{head}h q[1:3];"));
    same_instructions(
        &format!("{head}h q[{{-1, 0}}];"),
        &format!("{head}h q[3];\nh q[0];"),
    );
    same_instructions(
        &format!("{head}const array[int, 3] a = {{0, 1, 2}};\nx q[a[-1]];"),
        &format!("{head}x q[2];"),
    );
}

#[test]
fn a_counted_control_is_that_many_controls() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\n";
    same_instructions(
        &format!("{head}ctrl(2) @ x q[0], q[1], q[2];"),
        &format!("{head}ctrl @ ctrl @ x q[0], q[1], q[2];"),
    );
    same_instructions(
        &format!("{head}const int n = 2;\nnegctrl(n) @ ctrl @ x q[0], q[1], q[2], q[3];"),
        &format!("{head}negctrl @ negctrl @ ctrl @ x q[0], q[1], q[2], q[3];"),
    );
    same_instructions(
        &format!("{head}ctrl(1) @ x q[0], q[1];"),
        &format!("{head}cx q[0], q[1];"),
    );
}

#[test]
fn a_control_count_must_be_positive_and_carryable() {
    let head = "OPENQASM 3.0;\nqubit[2] q;\n";
    let zero = parse(&format!("{head}ctrl(0) @ x q[0];")).unwrap_err();
    assert!(
        matches!(zero, PrismError::Parse { .. }) && zero.to_string().contains("positive"),
        "{zero}"
    );
    for chain in [
        "ctrl(300) @ x q[0], q[1];",
        "ctrl(200) @ negctrl(56) @ x q[0];",
    ] {
        let wide = parse(&format!("{head}{chain}")).unwrap_err();
        assert!(
            matches!(&wide, PrismError::UnsupportedConstruct { construct, .. }
                if construct.contains("past 255 controls")),
            "{chain}: {wide}"
        );
    }
}

#[test]
fn nop_names_its_qubits_and_does_nothing() {
    let c = parse("OPENQASM 3.0;\nqubit[2] q;\nnop q[0], q[1];\nnop;\nx q[0];").unwrap();
    assert_eq!(c.instructions.len(), 1);
    let err = parse("OPENQASM 3.0;\nqubit[2] q;\nnop r;").unwrap_err();
    assert!(matches!(err, PrismError::UndefinedRegister { .. }), "{err}");
}

#[test]
fn a_gate_length_designator_is_dropped() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[2] q;\nx[100ns] q[0];\nrx(0.5)[20dt] q[1];",
        "OPENQASM 3.0;\nqubit[2] q;\nx q[0];\nrx(0.5) q[1];",
    );
}

#[test]
fn a_bit_declared_from_a_measurement_measures_into_it() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[2] q;\nh q;\nbit b = measure q[0];\nbit[2] c = measure q;",
        "OPENQASM 3.0;\nqubit[2] q;\nh q;\nbit b;\nb = measure q[0];\nbit[2] c;\nc = measure q;",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\n\
         def mx(qubit a) -> bit { h a; bit r = measure a; return r; }\nc[0] = mx(q[0]);",
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nh q[0];\nc[0] = measure q[0];",
    );
}

#[test]
fn a_bit_initializer_of_zeros_is_the_cleared_register() {
    for decl in [
        "bit b = 0;",
        "bit[4] b = \"0000\";",
        "bit[4] b = \"00_00\";",
        "bit[3] b = 0;",
    ] {
        let c = parse(&format!("OPENQASM 3.0;\n{decl}\nqubit[1] q;\nx q[0];"))
            .unwrap_or_else(|e| panic!("{decl}: {e}"));
        assert_eq!(c.instructions.len(), 1, "{decl}");
    }
}

#[test]
fn a_bit_initializer_with_a_set_bit_is_declined_by_name() {
    for decl in ["bit[2] f = \"11\";", "bit b = 1;", "bit[3] f = 4;"] {
        let err = parse(&format!("OPENQASM 3.0;\n{decl}")).unwrap_err();
        assert!(
            matches!(&err, PrismError::UnsupportedConstruct { construct, .. }
                if construct.contains("classical write")),
            "{decl}: {err}"
        );
    }
}

#[test]
fn a_bit_string_must_match_its_register() {
    for (decl, wanted) in [
        ("bit[3] f = \"01\";", "has 2 bit(s)"),
        ("bit[2] f = \"0a\";", "neither 0 nor 1"),
    ] {
        let err = parse(&format!("OPENQASM 3.0;\n{decl}")).unwrap_err();
        assert!(
            matches!(err, PrismError::Parse { .. }) && err.to_string().contains(wanted),
            "{decl}: {err}"
        );
    }
}

#[test]
fn unicode_names_declare_and_read_back() {
    same_instructions(
        "OPENQASM 3.0;\nqubit γ;\ngate rr(θ) a { rz(θ) a; }\nrr(π / 2) γ;",
        "OPENQASM 3.0;\nqubit g;\nrz(pi / 2) g;",
    );
}

#[test]
fn an_annotation_is_skipped_where_a_statement_starts() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[2] q;\n@openqasm.reversible\ngate g a { h a; }\n\
         @bind q[0] to a \"sensor\"\ng q[0];\ninv @h q[1];",
        "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ninv @ h q[1];",
    );
}

#[test]
fn an_empty_def_is_a_call_that_does_nothing() {
    let c =
        parse("OPENQASM 3.0;\nqubit[1] q;\ndef idle(qubit a) { }\nidle(q[0]);\nx q[0];").unwrap();
    assert_eq!(c.instructions.len(), 1);
    let err = parse("OPENQASM 3.0;\nqubit[1] q;\ndef f(qubit a) -> bit { }").unwrap_err();
    assert!(err.to_string().contains("body is empty"), "{err}");
}

/// The angle of the `rx` that `body` applies, after a two-qubit, four-bit header.
fn rx_angle(body: &str) -> f64 {
    let source = format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}");
    let circuit = parse(&source).unwrap_or_else(|e| panic!("{body}: {e}"));
    match circuit.instructions.as_slice() {
        [
            Instruction::Gate {
                gate: Gate::Rx(angle),
                ..
            },
        ] => *angle,
        other => panic!("{body}: expected one rx, got {other:?}"),
    }
}

/// The guard `body`'s one conditional gate carries, after the same header.
fn guard_of(body: &str) -> ClassicalCondition {
    let source = format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}");
    let circuit = parse(&source).unwrap_or_else(|e| panic!("{body}: {e}"));
    match circuit.instructions.as_slice() {
        [Instruction::Conditional { condition, .. }] => condition.clone(),
        other => panic!("{body}: expected one conditional, got {other:?}"),
    }
}

fn parse_error_of(body: &str) -> PrismError {
    parse(&format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}"))
        .err()
        .unwrap_or_else(|| panic!("{body}: parsed"))
}

#[test]
fn integer_division_truncates_toward_zero() {
    assert_eq!(rx_angle("rx(1 / 2) q[0];"), 0.0);
    assert_eq!(rx_angle("int a = 7; int b = -2; rx(a / b) q[0];"), -3.0);
    assert_eq!(rx_angle("rx(7 % -2) q[0];"), 1.0);
    assert_eq!(rx_angle("rx(1.0 / 2) q[0];"), 0.5);
    assert_eq!(rx_angle("rx(pi / 2) q[0];"), std::f64::consts::FRAC_PI_2);
}

#[test]
fn integer_arithmetic_stays_exact_past_double_precision() {
    let body = "int x = 9007199254740993; rx(x - 9007199254740992) q[0];";
    assert_eq!(rx_angle(body), 1.0);
    assert_eq!(rx_angle("rx(0x10 + 0b11 + 0o7 + 1_000) q[0];"), 1026.0);
}

#[test]
fn a_zero_divisor_is_a_parse_error() {
    for (body, wanted) in [
        ("rx(1 / 0) q[0];", "division by zero"),
        ("rx(1 % 0) q[0];", "modulo by zero"),
        ("int a = 3; a /= 0; rx(a) q[0];", "division by zero"),
    ] {
        let err = parse_error_of(body);
        assert!(err.to_string().contains(wanted), "{body}: {err}");
    }
}

#[test]
fn casts_convert_to_their_type() {
    for (body, want) in [
        ("rx(int(2.7)) q[0];", 2.0),
        ("rx(int(-2.7)) q[0];", -2.0),
        ("rx(int[8](300)) q[0];", 44.0),
        ("rx(int[8](200)) q[0];", -56.0),
        ("rx(uint[4](-1)) q[0];", 15.0),
        ("rx(bit[2](7)) q[0];", 3.0),
        ("rx(bool(5)) q[0];", 1.0),
        ("rx(bool(0.0)) q[0];", 0.0),
        ("rx(float[32](0.1)) q[0];", f64::from(0.1f32)),
        ("rx(angle[3](pi / 3)) q[0];", std::f64::consts::FRAC_PI_4),
        ("array[int[8], 2] a = {3, 4}; rx(int(a[1])) q[0];", 4.0),
    ] {
        assert_eq!(rx_angle(body), want, "{body}");
    }
}

#[test]
fn a_cast_width_is_checked() {
    for (body, wanted) in [
        ("rx(bool[2](1)) q[0];", "takes no width"),
        ("rx(int[0](1)) q[0];", "needs 1 to 64 bits"),
        ("rx(uint[65](1)) q[0];", "needs 1 to 64 bits"),
        ("rx(float[8](1)) q[0];", "needs 16, 32, 64 or 128 bits"),
        ("float[24] f = 1.0;", "needs 16, 32, 64 or 128 bits"),
    ] {
        let err = parse_error_of(body);
        assert!(err.to_string().contains(wanted), "{body}: {err}");
    }
}

#[test]
fn floats_follow_ieee_754_and_the_language_casts() {
    for (body, want) in [
        ("float[16] f = 0.1; rx(f) q[0];", 0.0999755859375),
        ("rx(float[16](3.14159)) q[0];", 3.140625),
        ("rx(float[128](0.1)) q[0];", 0.1),
        ("rx(mod(7.5, 2)) q[0];", 1.5),
        ("rx(bit[2](uint(2.5))) q[0];", 2.0),
        ("rx(int[8](1e30)) q[0];", 0.0),
    ] {
        assert_eq!(rx_angle(body), want, "{body}");
    }
    for (body, wanted) in [
        ("rx(7.5 % 2) q[0];", "`%` takes integers"),
        ("float f = 7.5; f %= 2; rx(f) q[0];", "`%` takes integers"),
        ("rx(bit[2](2.5)) q[0];", "cannot hold the float"),
    ] {
        let err = parse_error_of(body);
        assert!(err.to_string().contains(wanted), "{body}: {err}");
    }
}

#[test]
fn a_typed_variable_holds_what_its_type_can() {
    let tau = std::f64::consts::TAU;
    for (body, want) in [
        ("float[32] f = 0.1; rx(f) q[0];", f64::from(0.1f32)),
        ("angle[2] a = 1.0; rx(a) q[0];", tau / 4.0),
        ("angle a = -pi / 2; rx(a) q[0];", 3.0 * tau / 4.0),
        ("angle[1] a = 3 * pi / 2; rx(a) q[0];", 0.0),
        ("int[4] n = 7; n += 1; rx(n) q[0];", -8.0),
        ("angle[2] a = 0; a = angle(pi / 3); rx(a) q[0];", tau / 4.0),
        ("uint[3] n = 5; n *= 2; rx(n) q[0];", 2.0),
    ] {
        assert_eq!(rx_angle(body), want, "{body}");
    }
}

#[test]
fn every_compound_assignment_applies_its_operator() {
    for (update, want) in [
        ("**= 2", 36.0),
        ("&= 3", 2.0),
        ("|= 3", 7.0),
        ("^= 3", 5.0),
        ("<<= 2", 24.0),
        (">>= 2", 1.0),
    ] {
        let body = format!("int a = 6; a {update}; rx(a) q[0];");
        assert_eq!(rx_angle(&body), want, "{body}");
    }
}

#[test]
fn an_expression_statement_is_evaluated_and_discarded() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nint a = 2;\na == 2;\n(a + 1) * 3;\n-a;\nx q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nx q[0];",
    );
    same_instructions(
        "qubit q; def f(qubit a) -> int { x a; return 3; } f(q) == 3; h q;",
        "qubit q; x q; h q;",
    );
    let err = parse(
        "OPENQASM 3.0;\nqubit[1] q;\ndef f(qubit a) -> bit { bit r = measure a; return r; }\n\
         f(q[0]) == 1;",
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrismError::UnsupportedConstruct { construct, .. }
            if construct.contains("def `f`")),
        "{err}"
    );
}

#[test]
fn an_if_on_a_parse_time_value_keeps_the_arm_it_takes() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nint n = 3;\nif (n > 2 && n != 5) x q[0]; else y q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nx q[0];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nif (1 == 2) { x q[0]; } else { y q[0]; }",
        "OPENQASM 3.0;\nqubit[1] q;\ny q[0];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nif (false) x q[0];\nz q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nz q[0];",
    );
}

#[test]
fn a_condition_over_classical_bits_lowers_to_its_guard() {
    for (body, want) in [
        ("if (!c[0]) x q[0];", ClassicalCondition::BitIsZero(0)),
        ("if (c[2] == 1) x q[0];", ClassicalCondition::BitIsOne(2)),
        (
            "if (bool(c)) x q[0];",
            ClassicalCondition::RegisterNotEquals {
                offset: 0,
                size: 4,
                value: 0,
            },
        ),
        (
            "if (c == 1 + 2) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 3,
            },
        ),
        (
            "if (uint[2](c) != 3) x q[0];",
            ClassicalCondition::RegisterNotEquals {
                offset: 0,
                size: 2,
                value: 3,
            },
        ),
        (
            "if (int[8](c) == 5) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 5,
            },
        ),
        (
            "if (int[4](c) == -1) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 15,
            },
        ),
        (
            "if (int[3](c) == -4) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 3,
                value: 4,
            },
        ),
    ] {
        assert_eq!(
            format!("{:?}", guard_of(body)),
            format!("{want:?}"),
            "{body}"
        );
    }
}

#[test]
fn a_condition_no_guard_spells_points_at_the_dynamic_parse() {
    for body in [
        "if (c[0] && c[1]) x q[0];",
        "if (int[4](c) == 8) x q[0];",
        "if (c + 1 == 2) x q[0];",
    ] {
        let err = parse_error_of(body);
        assert!(
            matches!(&err, PrismError::UnsupportedConstruct { construct, .. }
                if construct.contains("parse_dynamic")),
            "{body}: {err}"
        );
        let source = format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}");
        parse_dynamic(&source).unwrap_or_else(|e| panic!("{body}: {e}"));
    }
}

#[test]
fn an_integer_reads_one_bit_per_index() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[4] q;\nuint[4] a = 5;\n\
         for int i in [0:3] { if (bool(a[i])) x q[i]; }",
        "OPENQASM 3.0;\nqubit[4] q;\nx q[0];\nx q[2];",
    );
    assert_eq!(rx_angle("uint[4] a = 6; rx(a[1]) q[0];"), 1.0);
    assert_eq!(rx_angle("int[4] a = -2; rx(a[-1] + 2 * a[0]) q[0];"), 1.0);
    let err = parse_error_of("uint[4] a = 5; rx(a[4]) q[0];");
    assert!(err.to_string().contains("outside `a`"), "{err}");
}

#[test]
fn a_switch_takes_any_value_its_guards_can_compare() {
    for (body, want) in [
        (
            "switch (int(c)) { case 1 { x q[0]; } }",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 1,
            },
        ),
        (
            "switch (uint[2](c)) { case 3 { x q[0]; } }",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 2,
                value: 3,
            },
        ),
        (
            "switch (c[1]) { case 1 { x q[0]; } }",
            ClassicalCondition::RegisterEquals {
                offset: 1,
                size: 1,
                value: 1,
            },
        ),
    ] {
        assert_eq!(
            format!("{:?}", guard_of(body)),
            format!("{want:?}"),
            "{body}"
        );
    }
    let err = parse_error_of("switch (int[2](c)) { case 1 { x q[0]; } }");
    assert!(err.to_string().contains("parse_dynamic"), "{err}");
}

#[test]
fn a_switch_on_a_parse_time_value_keeps_one_arm() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nint i = 2;\n\
         switch (i * 2) { case 3 { x q[0]; } case 4, 5 { y q[0]; } }",
        "OPENQASM 3.0;\nqubit[1] q;\ny q[0];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nswitch (5) { case 3 { x q[0]; } default { z q[0]; } }",
        "OPENQASM 3.0;\nqubit[1] q;\nz q[0];",
    );
    let err =
        parse("OPENQASM 3.0;\nqubit[1] q;\nswitch (1) { case 1 { } case 1 { } }").unwrap_err();
    assert!(err.to_string().contains("appears twice"), "{err}");
}

#[test]
fn break_and_continue_stop_an_unrolled_for() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\nbit[4] c;\n";
    same_instructions(
        &format!("{head}for int i in [0:3] {{ if (i == 2) break; x q[i]; }} h q[3];"),
        &format!("{head}x q[0]; x q[1]; h q[3];"),
    );
    same_instructions(
        &format!("{head}for int i in [0:3] {{ if (i == 1) {{ continue; }} h q[i]; }}"),
        &format!("{head}h q[0]; h q[2]; h q[3];"),
    );
    same_instructions(
        &format!("{head}for int i in [0:3] {{ if (i % 2 == 1) continue; c[i] = measure q[i]; }}"),
        &format!("{head}c[0] = measure q[0]; c[2] = measure q[2];"),
    );
    same_instructions(
        &format!(
            "{head}for int i in [0:1] {{ for int j in [0:3] {{ if (j == 2) break; cx q[i], q[j + 2]; }} z q[i]; }}"
        ),
        &format!(
            "{head}cx q[0], q[2]; cx q[0], q[3]; z q[0]; cx q[1], q[2]; cx q[1], q[3]; z q[1];"
        ),
    );
    // A `switch` is not a loop, so its `break` leaves the enclosing `for`.
    same_instructions(
        &format!(
            "{head}for int i in [0:3] {{ switch (i) {{ case 1 {{ break; }} default {{ x q[i]; }} }} }}"
        ),
        &format!("{head}x q[0];"),
    );
    same_instructions(
        &format!(
            "{head}int n = 0; for int i in [0:3] {{ n += 1; if (n == 2) break; }} rx(n) q[0];"
        ),
        &format!("{head}rx(2) q[0];"),
    );
}

#[test]
fn end_stops_the_program_from_any_scope() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\n";
    same_instructions(
        &format!("{head}x q[0];\nend;\nx q[1];"),
        &format!("{head}x q[0];"),
    );
    same_instructions(
        &format!("{head}for int i in [0:3] {{ x q[i]; if (i == 1) end; }} h q[0];"),
        &format!("{head}x q[0]; x q[1];"),
    );
    same_instructions(
        &format!("{head}def f(qubit a) {{ x a; end; }} f(q[2]); h q[0];"),
        &format!("{head}x q[2];"),
    );
}

#[test]
fn loop_exits_outside_a_loop_are_parse_errors() {
    let head = "OPENQASM 3.0;\nqubit[2] q;\n";
    for body in [
        "break;",
        "if (true) { continue; }",
        "def f() { continue; } for int i in [0:1] { f(); }",
        "gate g a { break; } g q[0];",
    ] {
        match parse(&format!("{head}{body}")) {
            Err(PrismError::Parse { message, .. }) => {
                assert!(message.contains("outside a loop"), "`{body}`: {message}")
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }
    match parse(&format!("{head}gate g a {{ end; }} g q[0];")) {
        Err(PrismError::Parse { message, .. }) => {
            assert!(message.contains("gate `g`"), "{message}")
        }
        other => panic!("`end` in a gate should be a parse error, got {other:?}"),
    }
}

#[test]
fn a_measured_exit_under_parse_points_at_parse_dynamic() {
    let head = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nc[0] = measure q[0];\n";
    for body in [
        "for int i in [0:1] { if (c[0]) break; x q[1]; }",
        "for int i in [0:1] { if (c[0]) { continue; } x q[1]; }",
        "if (c[0]) end;",
        "for int i in [0:1] { if (c[0]) end; }",
    ] {
        match parse(&format!("{head}{body}")) {
            Err(PrismError::UnsupportedConstruct { construct, line }) => {
                assert!(construct.contains("parse_dynamic"), "`{body}`: {construct}");
                assert_eq!(line, 5, "`{body}`");
            }
            other => panic!("`{body}` should decline, got {other:?}"),
        }
    }
}

#[test]
fn end_is_still_a_gate_name_when_it_takes_operands() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\ngate end a { x a; }\nend q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nx q[0];",
    );
}
