//! Gate resolution, modifiers, broadcasting, registers and source syntax.

use super::*;
use crate::test_support::assert_mat_close;

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
fn malformed_programs_are_rejected_with_their_error_kind() {
    type Kind = fn(&PrismError) -> bool;
    let parse_error: Kind = |e| matches!(e, PrismError::Parse { .. });
    let arity: Kind = |e| matches!(e, PrismError::GateArity { .. });
    let unsupported: Kind = |e| matches!(e, PrismError::UnsupportedConstruct { .. });
    let undefined: Kind = |e| matches!(e, PrismError::UndefinedRegister { .. });
    let bad_qubit: Kind = |e| matches!(e, PrismError::InvalidQubit { .. });
    let cases: [(&str, &str, Kind); 25] = [
        (
            "test_unsupported_def_with_measurement",
            "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\ndef mygate(qubit q) { measure q -> c[0]; }",
            unsupported,
        ),
        (
            "test_undefined_register",
            "OPENQASM 3.0;\nh q[0];",
            undefined,
        ),
        (
            "test_gate_def_missing_name_is_error",
            "OPENQASM 3.0;\ngate {}",
            parse_error,
        ),
        (
            "test_qubit_out_of_bounds",
            "OPENQASM 3.0;\nqubit[2] q;\nh q[5];",
            bad_qubit,
        ),
        (
            "test_gate_arity_mismatch",
            "OPENQASM 3.0;\nqubit[3] q;\ncx q[0], q[1], q[2];",
            arity,
        ),
        (
            "test_cp_arity_mismatch",
            "OPENQASM 3.0;\nqubit[1] q;\ncp(pi/4) q[0];",
            arity,
        ),
        (
            "test_broadcast_register_size_mismatch",
            r#"
        OPENQASM 3.0;
        qubit[2] q;
        qubit[3] r;
        cx q, r;
    "#,
            parse_error,
        ),
        (
            "test_if_missing_body",
            "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0])",
            parse_error,
        ),
        (
            "test_if_undefined_creg",
            "OPENQASM 3.0;\nqubit[1] q;\nif(c==0) x q[0];",
            undefined,
        ),
        (
            "test_gate_def_arity_mismatch",
            r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate mygate a, b { cx a, b; }
        mygate q[0];
    "#,
            arity,
        ),
        (
            "test_missing_bracket_qubit_decl",
            "OPENQASM 3.0;\nqubit[4 q;",
            parse_error,
        ),
        (
            "test_missing_bracket_bit_decl",
            "OPENQASM 3.0;\nbit[4 c;",
            parse_error,
        ),
        (
            "test_pi_div_zero",
            "OPENQASM 3.0;\nqubit[1] q;\nrz(pi/0) q[0];",
            parse_error,
        ),
        (
            "test_neg_pi_div_zero",
            "OPENQASM 3.0;\nqubit[1] q;\nrz(-pi/0) q[0];",
            parse_error,
        ),
        (
            "test_nan_angle_rejected",
            "OPENQASM 3.0;\nqubit[1] q;\nrz(NaN) q[0];",
            parse_error,
        ),
        (
            "test_inf_angle_rejected",
            "OPENQASM 3.0;\nqubit[1] q;\nrz(inf) q[0];",
            parse_error,
        ),
        (
            "test_recursive_gate_def_rejected",
            r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate loop a { loop a; }
        loop q[0];
    "#,
            parse_error,
        ),
        (
            "for_loop_unterminated_brace_rejected",
            r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:2] {
            x q[i];
    "#,
            parse_error,
        ),
        (
            "for_loop_zero_step_rejected",
            r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:0:2] {
            x q[i];
        }
    "#,
            parse_error,
        ),
        (
            "for_loop_non_integer_bound_rejected",
            r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:pi] {
            x q[i];
        }
    "#,
            parse_error,
        ),
        (
            "def_returning_a_literal_is_rejected",
            r#"
        OPENQASM 3.0;
        qubit[1] q;
        def measure_qubit(qubit a) -> bit {
            return 0;
        }
    "#,
            unsupported,
        ),
        (
            "def_subroutine_arity_mismatch_rejected",
            r#"
        OPENQASM 3.0;
        qubit[2] q;
        def bell(qubit a, qubit b) {
            h a;
            cx a, b;
        }
        bell(q[0]);
    "#,
            arity,
        ),
        (
            "def_subroutine_recursion_capped",
            r#"
        OPENQASM 3.0;
        qubit[1] q;
        def loop(qubit a) {
            loop(a);
        }
        loop(q[0]);
    "#,
            parse_error,
        ),
        (
            "if_negative_value_rejected",
            r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[4] c;
        if (c == -1) x q[0];
    "#,
            parse_error,
        ),
        (
            "if_bit_compare_against_two_rejected",
            r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] == 2) x q[0];
    "#,
            parse_error,
        ),
    ];
    for (name, source, kind) in cases {
        match parse(source) {
            Ok(_) => panic!("{name}: accepted"),
            Err(err) => assert!(kind(&err), "{name}: {err:?}"),
        }
    }
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
            assert_mat_close(mat, &Gate::H.matrix_2x2(), 1e-12);
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
            assert_mat_close(mat, &expected, 1e-12);
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
                assert_mat_close(mat, &expected_mat, 1e-12);
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
            assert_mat_close(mat, &expected, 1e-12);
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
                assert_mat_close(&data.mat, &Gate::X.matrix_2x2(), 1e-12);
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
            plus_s * num_complex::Complex64::from_polar(1.0, plus_beta),
            zero,
        ],
        [
            zero,
            plus_s * num_complex::Complex64::from_polar(1.0, -plus_beta),
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
            assert_mat_close(mat, &expected_plus, 1e-12);
        } else {
            panic!("expected Fused2q");
        }
    } else {
        panic!("expected gate");
    }
    if let Instruction::Gate { gate, targets } = &c.instructions[1] {
        assert_eq!(targets.as_slice(), &[0, 1]);
        if let Gate::Fused2q(mat) = gate {
            assert_mat_close(mat, &expected_minus, 1e-12);
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
                assert_mat_close(mat, &expected_mat, 1e-12);
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
    assert_mat_close(gpi, &[[zero, -i], [i, zero]], 1e-12);

    let Instruction::Gate {
        gate: Gate::Fused2q(ms),
        ..
    } = &c.instructions[1]
    else {
        panic!("expected a Fused2q ms");
    };
    let h = num_complex::Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let o = num_complex::Complex64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2);
    assert_mat_close(
        ms,
        &[
            [h, zero, zero, o],
            [zero, h, o, zero],
            [zero, o, h, zero],
            [o, zero, zero, h],
        ],
        1e-12,
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
        let resolved = match super::super::pauli_rotation_axes(gate.name()) {
            Some(axes) => {
                let factors: Vec<PauliTerm> = axes
                    .iter()
                    .enumerate()
                    .map(|(q, &axis)| PauliTerm::new(q, axis))
                    .collect();
                super::super::pauli_rotation_gate(params[0], &factors).0
            }
            None => gate_library::resolve_gate(gate.name(), &params, Dialect::Native, 0)
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
fn xx_plus_yy_matches_rz_conjugation_on_first_target() {
    let theta = 0.7;
    let beta = 0.4;
    let qasm = format!("OPENQASM 3.0;\nqubit[2] q;\nxx_plus_yy({theta}, {beta}) q[0], q[1];\n");
    let conjugated = format!(
        "OPENQASM 3.0;\nqubit[2] q;\nrz({beta}) q[0];\nxx_plus_yy({theta}, 0) q[0], q[1];\nrz(-{beta}) q[0];\n"
    );
    let direct_c = parse(&qasm).unwrap();
    let conj_c = parse(&conjugated).unwrap();
    let mut b1 = crate::backend::statevector::StatevectorBackend::new(42);
    let mut b2 = crate::backend::statevector::StatevectorBackend::new(42);
    crate::sim::run_on(
        &mut b1,
        &parse("OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nx q[1];\n").unwrap(),
    )
    .unwrap();
    crate::sim::run_on(
        &mut b2,
        &parse("OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nx q[1];\n").unwrap(),
    )
    .unwrap();
    use crate::backend::Backend;
    for inst in &direct_c.instructions {
        b1.apply(inst).unwrap();
    }
    for inst in &conj_c.instructions {
        b2.apply(inst).unwrap();
    }
    for (a, b) in b1.state_vector().iter().zip(b2.state_vector().iter()) {
        assert!((a - b).norm() < 1e-12, "{a} vs {b}");
    }
}

#[test]
fn user_gate_shadows_decomposed_builtins_and_composes_with_modifiers() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[2] q;\ngate cswap a, b { cx a, b; }\ncswap q[0], q[1];",
        "OPENQASM 3.0;\nqubit[2] q;\ncx q[0], q[1];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[2] q;\ngate u3(a) q { rx(a) q; }\ninv @ u3(0.4) q[0];\npow(2) @ u3(0.3) q[1];",
        "OPENQASM 3.0;\nqubit[2] q;\ninv @ rx(0.4) q[0];\nrx(0.3) q[1];\nrx(0.3) q[1];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[4] q;\ngate iswap a, b { rz(0.2) a; cz a, b; }\nctrl @ negctrl @ iswap q[0], q[1], q[2], q[3];",
        "OPENQASM 3.0;\nqubit[4] q;\ngate my_iswap a, b { rz(0.2) a; cz a, b; }\nctrl @ negctrl @ my_iswap q[0], q[1], q[2], q[3];",
    );
    let (tagged, _) = crate::circuit::openqasm::parse_parametric(
        "OPENQASM 3.0;\ninput float t;\nqubit[1] q;\ngate u3(a) q { rz(a) q; }\nu3(t) q[0];",
    )
    .unwrap();
    assert_eq!(tagged.instructions.len(), 1);
}

#[test]
fn gphase_power_overflow_is_rejected_without_nan() {
    for source in [
        "OPENQASM 3.0;\nqubit[1] q;\npow(2) @ gphase(1e308);",
        "OPENQASM 3.0;\nqubit[1] q;\npow(0) @ pow(2) @ gphase(1e308);",
        "OPENQASM 3.0;\nqubit[1] q;\nctrl @ pow(2) @ gphase(1e308) q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\npow(1e6) @ gphase(1e303);",
        "OPENQASM 3.0;\nqubit[1] q;\npow(999999.5) @ pow(999999.5) @ gphase(1e297);",
    ] {
        let err = parse(source).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }), "{source}: {err:?}");
    }
    for source in [
        "OPENQASM 3.0;\ninput float t;\nqubit[1] q;\npow(2) @ gphase(t + 1e308);",
        "OPENQASM 3.0;\ninput float t;\nqubit[1] q;\npow(0) @ pow(2) @ gphase(t + 1e308);",
    ] {
        let err = crate::circuit::openqasm::parse_parametric(source).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }), "{source}: {err:?}");
    }
}
