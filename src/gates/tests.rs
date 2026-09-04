use super::*;

#[test]
fn format_angle_pi_fractions() {
    assert_eq!(format_angle(std::f64::consts::PI), "π");
    assert_eq!(format_angle(std::f64::consts::FRAC_PI_2), "π/2");
    assert_eq!(format_angle(std::f64::consts::FRAC_PI_4), "π/4");
    assert_eq!(format_angle(-std::f64::consts::FRAC_PI_4), "-π/4");
    assert_eq!(format_angle(std::f64::consts::PI / 3.0), "π/3");
    assert_eq!(format_angle(0.123), "0.1230");
}

#[test]
fn display_labels() {
    assert_eq!(Gate::H.to_string(), "H");
    assert_eq!(Gate::Cx.to_string(), "CX");
    assert_eq!(Gate::Rx(std::f64::consts::FRAC_PI_2).to_string(), "Rx(π/2)");
    assert_eq!(Gate::Rz(0.5).to_string(), "Rz(0.5000)");
    assert_eq!(Gate::Id.to_string(), "I");
    assert_eq!(Gate::Swap.to_string(), "SWAP");
}

#[test]
fn test_gate_arity() {
    assert_eq!(Gate::H.num_qubits(), 1);
    assert_eq!(Gate::Rx(0.5).num_qubits(), 1);
    assert_eq!(Gate::Cx.num_qubits(), 2);
    assert_eq!(Gate::Swap.num_qubits(), 2);
}

#[test]
fn batch_gate_arity_counts_qubits_above_word_boundary() {
    let one = Complex64::new(1.0, 0.0);
    let batch_rzz = Gate::BatchRzz(Box::new(BatchRzzData {
        edges: vec![(0, 64, 0.25), (64, 129, 0.5)],
    }));
    assert_eq!(batch_rzz.num_qubits(), 3);

    let diagonal_batch = Gate::DiagonalBatch(Box::new(DiagonalBatchData {
        entries: vec![
            DiagEntry::Phase1q {
                qubit: 64,
                d0: one,
                d1: -one,
            },
            DiagEntry::Phase2q {
                q0: 64,
                q1: 130,
                phase: -one,
            },
        ],
    }));
    assert_eq!(diagonal_batch.num_qubits(), 2);

    let multi_2q = Gate::Multi2q(Box::new(Multi2qData {
        gates: vec![
            (63, 64, Gate::Cx.matrix_4x4()),
            (64, 130, Gate::Cz.matrix_4x4()),
        ],
    }));
    assert_eq!(multi_2q.num_qubits(), 3);
}

#[test]
fn test_h_matrix_is_unitary() {
    let m = Gate::H.matrix_2x2();
    // H * H = I
    let mut product = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for (k, row) in m.iter().enumerate() {
                product[i][j] += m[i][k] * row[j];
            }
        }
    }
    let eps = 1e-12;
    assert!((product[0][0].re - 1.0).abs() < eps);
    assert!(product[0][0].im.abs() < eps);
    assert!(product[0][1].norm() < eps);
    assert!(product[1][0].norm() < eps);
    assert!((product[1][1].re - 1.0).abs() < eps);
}

#[test]
fn test_rx_pi_equals_neg_i_x() {
    let rx = Gate::Rx(std::f64::consts::PI).matrix_2x2();
    // Rx(π) = -i·X  (up to global phase)
    // |Rx(π)[0][1]| should be 1
    assert!((rx[0][1].norm() - 1.0).abs() < 1e-12);
    assert!((rx[1][0].norm() - 1.0).abs() < 1e-12);
    assert!(rx[0][0].norm() < 1e-12);
    assert!(rx[1][1].norm() < 1e-12);
}

#[test]
fn test_clifford_classification() {
    assert!(Gate::H.is_clifford());
    assert!(Gate::S.is_clifford());
    assert!(Gate::Cx.is_clifford());
    assert!(!Gate::T.is_clifford());
    assert!(!Gate::Rx(0.5).is_clifford());
    assert!(!Gate::Cu(Box::new([[Complex64::new(1.0, 0.0); 2]; 2])).is_clifford());
}

#[test]
fn test_preserves_sparsity() {
    // Diagonal and permutation gates preserve sparsity
    assert!(Gate::Id.preserves_sparsity());
    assert!(Gate::X.preserves_sparsity());
    assert!(Gate::Y.preserves_sparsity());
    assert!(Gate::Z.preserves_sparsity());
    assert!(Gate::S.preserves_sparsity());
    assert!(Gate::T.preserves_sparsity());
    assert!(Gate::Rz(1.0).preserves_sparsity());
    assert!(Gate::P(0.5).preserves_sparsity());
    assert!(Gate::Cx.preserves_sparsity());
    assert!(Gate::Cz.preserves_sparsity());
    assert!(Gate::Swap.preserves_sparsity());

    // Superposition-creating gates do NOT preserve sparsity
    assert!(!Gate::H.preserves_sparsity());
    assert!(!Gate::Rx(0.5).preserves_sparsity());
    assert!(!Gate::Ry(0.5).preserves_sparsity());
    assert!(!Gate::SX.preserves_sparsity());
    assert!(!Gate::SXdg.preserves_sparsity());

    // Cu with diagonal matrix preserves sparsity
    let diag = Box::new([
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ]);
    assert!(Gate::Cu(diag).preserves_sparsity());

    // Cu with H-like matrix does NOT preserve sparsity
    let h_mat = Box::new(Gate::H.matrix_2x2());
    assert!(!Gate::Cu(h_mat).preserves_sparsity());
}

#[test]
fn test_cu_arity() {
    let mat = Gate::H.matrix_2x2();
    assert_eq!(Gate::Cu(Box::new(mat)).num_qubits(), 2);
}

fn assert_mat_close(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2], eps: f64) {
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (a[i][j] - b[i][j]).norm() < eps,
                "mat[{i}][{j}]: expected {:?}, got {:?}",
                b[i][j],
                a[i][j]
            );
        }
    }
}

#[test]
fn test_inverse_self_inverse() {
    assert_eq!(Gate::H.inverse(), Gate::H);
    assert_eq!(Gate::X.inverse(), Gate::X);
    assert_eq!(Gate::Y.inverse(), Gate::Y);
    assert_eq!(Gate::Z.inverse(), Gate::Z);
    assert_eq!(Gate::Id.inverse(), Gate::Id);
    assert_eq!(Gate::Cx.inverse(), Gate::Cx);
    assert_eq!(Gate::Cz.inverse(), Gate::Cz);
    assert_eq!(Gate::Swap.inverse(), Gate::Swap);
}

#[test]
fn test_inverse_adjoint_pairs() {
    assert_eq!(Gate::S.inverse(), Gate::Sdg);
    assert_eq!(Gate::Sdg.inverse(), Gate::S);
    assert_eq!(Gate::T.inverse(), Gate::Tdg);
    assert_eq!(Gate::Tdg.inverse(), Gate::T);
}

#[test]
fn test_inverse_parametric() {
    assert_eq!(Gate::Rx(0.5).inverse(), Gate::Rx(-0.5));
    assert_eq!(Gate::Ry(1.0).inverse(), Gate::Ry(-1.0));
    assert_eq!(Gate::Rz(PI).inverse(), Gate::Rz(-PI));
}

#[test]
fn test_inverse_fused_is_adjoint() {
    let s_mat = Gate::S.matrix_2x2();
    let fused = Gate::Fused(Box::new(s_mat));
    let inv = fused.inverse();
    if let Gate::Fused(inv_mat) = &inv {
        assert_mat_close(inv_mat, &Gate::Sdg.matrix_2x2(), 1e-12);
    } else {
        panic!("expected Fused");
    }
}

#[test]
fn test_inverse_cu() {
    let rz_mat = Gate::Rz(0.5).matrix_2x2();
    let cu = Gate::Cu(Box::new(rz_mat));
    let inv = cu.inverse();
    if let Gate::Cu(inv_mat) = &inv {
        let expected = Gate::Rz(-0.5).matrix_2x2();
        assert_mat_close(inv_mat, &expected, 1e-12);
    } else {
        panic!("expected Cu");
    }
}

#[test]
fn test_matrix_power_zero() {
    assert_eq!(Gate::X.matrix_power(0), Gate::Id);
    assert_eq!(Gate::Rz(0.5).matrix_power(0), Gate::Id);
}

#[test]
fn test_matrix_power_one() {
    assert_eq!(Gate::X.matrix_power(1), Gate::X);
    assert_eq!(Gate::H.matrix_power(1), Gate::H);
}

#[test]
fn test_matrix_power_x_squared() {
    let x2 = Gate::X.matrix_power(2);
    if let Gate::Fused(mat) = &x2 {
        assert_mat_close(mat, &Gate::Id.matrix_2x2(), 1e-12);
    } else {
        panic!("expected Fused");
    }
}

#[test]
fn test_matrix_power_t_squared_is_s() {
    let t2 = Gate::T.matrix_power(2);
    if let Gate::Fused(mat) = &t2 {
        assert_mat_close(mat, &Gate::S.matrix_2x2(), 1e-12);
    } else {
        panic!("expected Fused");
    }
}

#[test]
fn test_matrix_power_negative() {
    let t_inv2 = Gate::T.matrix_power(-2);
    if let Gate::Fused(mat) = &t_inv2 {
        assert_mat_close(mat, &Gate::Sdg.matrix_2x2(), 1e-12);
    } else {
        panic!("expected Fused");
    }
}

#[test]
fn test_mcu_arity() {
    let mat = Gate::H.matrix_2x2();
    let mcu2 = Gate::Mcu(Box::new(McuData {
        mat,
        num_controls: 2,
    }));
    assert_eq!(mcu2.num_qubits(), 3);
    let mcu3 = Gate::Mcu(Box::new(McuData {
        mat,
        num_controls: 3,
    }));
    assert_eq!(mcu3.num_qubits(), 4);
}

#[test]
fn test_mcu_not_clifford() {
    let mat = Gate::X.matrix_2x2();
    let mcu = Gate::Mcu(Box::new(McuData {
        mat,
        num_controls: 2,
    }));
    assert!(!mcu.is_clifford());
}

#[test]
fn test_mcu_inverse() {
    let rz_mat = Gate::Rz(0.5).matrix_2x2();
    let mcu = Gate::Mcu(Box::new(McuData {
        mat: rz_mat,
        num_controls: 2,
    }));
    let inv = mcu.inverse();
    if let Gate::Mcu(inv_data) = &inv {
        let expected = Gate::Rz(-0.5).matrix_2x2();
        assert_mat_close(&inv_data.mat, &expected, 1e-12);
        assert_eq!(inv_data.num_controls, 2);
    } else {
        panic!("expected Mcu");
    }
}

#[test]
fn test_mcu_name() {
    let mat = Gate::H.matrix_2x2();
    let mcu = Gate::Mcu(Box::new(McuData {
        mat,
        num_controls: 2,
    }));
    assert_eq!(mcu.name(), "mcu");
}

#[test]
fn test_cphase_constructor() {
    let g = Gate::cphase(PI / 4.0);
    assert_eq!(g.num_qubits(), 2);
    assert_eq!(g.name(), "cu");
    if let Gate::Cu(mat) = &g {
        let one = Complex64::new(1.0, 0.0);
        assert!((mat[0][0] - one).norm() < 1e-14);
        assert!(mat[0][1].norm() < 1e-14);
        assert!(mat[1][0].norm() < 1e-14);
        let expected = Complex64::from_polar(1.0, PI / 4.0);
        assert!((mat[1][1] - expected).norm() < 1e-14);
    } else {
        panic!("expected Cu");
    }
}

#[test]
fn test_controlled_phase_detection() {
    let cp = Gate::cphase(0.5);
    assert!(cp.controlled_phase().is_some());
    let phase = cp.controlled_phase().unwrap();
    let expected = Complex64::from_polar(1.0, 0.5);
    assert!((phase - expected).norm() < 1e-14);

    // Non-diagonal Cu should not be detected
    let h_mat = Gate::H.matrix_2x2();
    let cu_h = Gate::Cu(Box::new(h_mat));
    assert!(cu_h.controlled_phase().is_none());

    // CZ is Cu([[1,0],[0,-1]]), should be detected (phase = -1)
    let z_mat = Gate::Z.matrix_2x2();
    let cu_z = Gate::Cu(Box::new(z_mat));
    assert!(cu_z.controlled_phase().is_some());
    let z_phase = cu_z.controlled_phase().unwrap();
    assert!((z_phase.re - (-1.0)).abs() < 1e-14);

    // Rz-based Cu is diagonal but mat[0][0] != 1, should NOT be detected
    let rz_mat = Gate::Rz(0.5).matrix_2x2();
    let cu_rz = Gate::Cu(Box::new(rz_mat));
    assert!(cu_rz.controlled_phase().is_none());

    // Non-Cu gates should return None
    assert!(Gate::H.controlled_phase().is_none());
    assert!(Gate::Cx.controlled_phase().is_none());
}

#[test]
fn test_controlled_phase_mcu() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let phase = Complex64::from_polar(1.0, 0.7);
    let mcu = Gate::Mcu(Box::new(McuData {
        mat: [[one, zero], [zero, phase]],
        num_controls: 2,
    }));
    assert!(mcu.controlled_phase().is_some());
    assert!((mcu.controlled_phase().unwrap() - phase).norm() < 1e-14);
}

#[test]
fn test_sx_matrix_is_sqrt_x() {
    let sx = Gate::SX.matrix_2x2();
    let sx2 = mat_mul_2x2(&sx, &sx);
    assert_mat_close(&sx2, &Gate::X.matrix_2x2(), 1e-12);
}

#[test]
fn test_sxdg_is_sx_inverse() {
    let sx = Gate::SX.matrix_2x2();
    let sxdg = Gate::SXdg.matrix_2x2();
    let product = mat_mul_2x2(&sx, &sxdg);
    assert_mat_close(&product, &Gate::Id.matrix_2x2(), 1e-12);
}

#[test]
fn test_p_gate_matrix() {
    let p = Gate::P(PI / 4.0).matrix_2x2();
    let t = Gate::T.matrix_2x2();
    assert_mat_close(&p, &t, 1e-12);
}

#[test]
fn test_sx_is_clifford() {
    assert!(Gate::SX.is_clifford());
    assert!(Gate::SXdg.is_clifford());
}

#[test]
fn test_p_inverse() {
    assert_eq!(Gate::P(0.5).inverse(), Gate::P(-0.5));
}

#[test]
fn test_sx_inverse_pair() {
    assert_eq!(Gate::SX.inverse(), Gate::SXdg);
    assert_eq!(Gate::SXdg.inverse(), Gate::SX);
}

#[test]
fn test_is_diagonal_1q() {
    assert!(Gate::Id.is_diagonal_1q());
    assert!(Gate::Z.is_diagonal_1q());
    assert!(Gate::S.is_diagonal_1q());
    assert!(Gate::Sdg.is_diagonal_1q());
    assert!(Gate::T.is_diagonal_1q());
    assert!(Gate::Tdg.is_diagonal_1q());
    assert!(Gate::Rz(0.5).is_diagonal_1q());
    assert!(Gate::P(0.5).is_diagonal_1q());
    assert!(!Gate::H.is_diagonal_1q());
    assert!(!Gate::X.is_diagonal_1q());
    assert!(!Gate::Y.is_diagonal_1q());
    assert!(!Gate::Rx(0.5).is_diagonal_1q());
    assert!(!Gate::Ry(0.5).is_diagonal_1q());
    assert!(!Gate::SX.is_diagonal_1q());
    assert!(!Gate::Cx.is_diagonal_1q());

    let diag_fused = Gate::Fused(Box::new(Gate::T.matrix_2x2()));
    assert!(diag_fused.is_diagonal_1q());
    let nondiag_fused = Gate::Fused(Box::new(Gate::H.matrix_2x2()));
    assert!(!nondiag_fused.is_diagonal_1q());
}

#[test]
fn test_is_self_inverse_2q() {
    assert!(Gate::Cx.is_self_inverse_2q());
    assert!(Gate::Cz.is_self_inverse_2q());
    assert!(Gate::Swap.is_self_inverse_2q());
    assert!(!Gate::H.is_self_inverse_2q());
    assert!(!Gate::T.is_self_inverse_2q());
    let mat = Gate::H.matrix_2x2();
    assert!(!Gate::Cu(Box::new(mat)).is_self_inverse_2q());
}

#[test]
fn test_gate_enum_size() {
    assert_eq!(
        std::mem::size_of::<Gate>(),
        16,
        "Gate enum must stay at 16 bytes"
    );
}

#[test]
fn test_recognize_named_gates() {
    for gate in &[
        Gate::H,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::SX,
        Gate::SXdg,
    ] {
        let mat = gate.matrix_2x2();
        let recognized = Gate::recognize_matrix(&mat);
        assert_eq!(
            recognized.as_ref(),
            Some(gate),
            "failed to recognize {:?}",
            gate.name()
        );
    }
}

#[test]
fn test_recognize_identity() {
    let id = Gate::Id.matrix_2x2();
    assert_eq!(Gate::recognize_matrix(&id), Some(Gate::Id));
}

#[test]
fn test_recognize_t_squared_is_s() {
    let t = Gate::T.matrix_2x2();
    let tt = mat_mul_2x2(&t, &t);
    assert_eq!(Gate::recognize_matrix(&tt), Some(Gate::S));
}

#[test]
fn test_recognize_s_squared_is_z() {
    let s = Gate::S.matrix_2x2();
    let ss = mat_mul_2x2(&s, &s);
    assert_eq!(Gate::recognize_matrix(&ss), Some(Gate::Z));
}

#[test]
fn test_recognize_h_squared_is_identity() {
    let h = Gate::H.matrix_2x2();
    let hh = mat_mul_2x2(&h, &h);
    assert_eq!(Gate::recognize_matrix(&hh), Some(Gate::Id));
}

#[test]
fn test_recognize_t_fourth_is_z() {
    let t = Gate::T.matrix_2x2();
    let t2 = mat_mul_2x2(&t, &t);
    let t4 = mat_mul_2x2(&t2, &t2);
    assert_eq!(Gate::recognize_matrix(&t4), Some(Gate::Z));
}

#[test]
fn test_recognize_non_clifford_returns_none() {
    let rx = Gate::Rx(0.7).matrix_2x2();
    assert_eq!(Gate::recognize_matrix(&rx), None);
    let ry = Gate::Ry(1.3).matrix_2x2();
    assert_eq!(Gate::recognize_matrix(&ry), None);
}

// A named gate carries no scalar, so a phased match has to fail: emitting `H`
// for `e^{i0.42}·H` would drop the factor. `T·X·T` is the product a fusion run
// actually builds, and it is `e^{iπ/4}·X`.
#[test]
fn test_recognize_rejects_a_global_phase() {
    let phase = Complex64::from_polar(1.0, 0.42);
    let h = Gate::H.matrix_2x2();
    let phased = [
        [h[0][0] * phase, h[0][1] * phase],
        [h[1][0] * phase, h[1][1] * phase],
    ];
    assert_eq!(Gate::recognize_matrix(&phased), None);

    let t = Gate::T.matrix_2x2();
    let txt = mat_mul_2x2(&t, &mat_mul_2x2(&Gate::X.matrix_2x2(), &t));
    assert_eq!(Gate::recognize_matrix(&txt), None);
}
