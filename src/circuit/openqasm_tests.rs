use super::*;

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
fn test_unsupported_gate_def() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\ndef mygate(qubit q) { x q; }";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::UnsupportedConstruct { .. }));
}

#[test]
fn test_undefined_register() {
    let qasm = "OPENQASM 3.0;\nh q[0];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::UndefinedRegister { .. }));
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
fn test_ctrl_swap_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ swap q[0], q[1], q[2];";
    let err = parse(qasm).unwrap_err();
    assert!(matches!(err, PrismError::UnsupportedConstruct { .. }));
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

#[test]
fn test_rxx_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nrxx(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 7);
}

#[test]
fn test_ryy_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nryy(pi/4) q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 7);
}

#[test]
fn test_iswap_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\niswap q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 6);
}

#[test]
fn test_ecr_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\necr q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.instructions.len(), 4);
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
fn test_empty_gate_body() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate noop a { }
        noop q[0];
    "#;
    let err = parse(qasm);
    assert!(err.is_err(), "empty gate body should be rejected");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("empty body"),
        "error should mention empty body: {msg}"
    );
}

#[test]
fn test_register_name_collision() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nqubit[2] q;";
    let err = parse(qasm);
    // Either error or silently overwrite — both acceptable, just don't panic
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
fn test_expr_e_constant() {
    assert!((eval_expr("e", 0, None).unwrap() - std::f64::consts::E).abs() < 1e-12);
    assert!((eval_expr("euler", 0, None).unwrap() - std::f64::consts::E).abs() < 1e-12);
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
    vars.insert("theta".to_string(), std::f64::consts::FRAC_PI_4);
    vars.insert("phi".to_string(), std::f64::consts::FRAC_PI_2);
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
fn test_replace_word_boundary() {
    assert_eq!(replace_word("theta a", "a", "X"), "theta X");
    assert_eq!(replace_word("a theta a", "a", "X"), "X theta X");
    assert_eq!(replace_word("abc a ab", "a", "X"), "abc X ab");
    assert_eq!(
        replace_word("rz(theta) a", "a", "__q__[0]"),
        "rz(theta) __q__[0]"
    );
}

#[test]
fn test_split_top_level_commas_nested() {
    let parts = split_top_level_commas("sin(pi/4), cos(0)");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].trim(), "sin(pi/4)");
    assert_eq!(parts[1].trim(), "cos(0)");
}

#[test]
fn test_split_top_level_commas_simple() {
    let parts = split_top_level_commas("pi/2, pi/4, 0.1");
    assert_eq!(parts.len(), 3);
}
