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
    // Rx(π) = -i·X
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

/// Position of `gate`'s variant in declaration order. No wildcard arm, so a new variant
/// fails to compile here until `one_of_each_variant` lists it.
fn variant_index(gate: &Gate) -> usize {
    match gate {
        Gate::Id => 0,
        Gate::X => 1,
        Gate::Y => 2,
        Gate::Z => 3,
        Gate::H => 4,
        Gate::S => 5,
        Gate::Sdg => 6,
        Gate::T => 7,
        Gate::Tdg => 8,
        Gate::SX => 9,
        Gate::SXdg => 10,
        Gate::Rx(_) => 11,
        Gate::Ry(_) => 12,
        Gate::Rz(_) => 13,
        Gate::P(_) => 14,
        Gate::Rzz(_) => 15,
        Gate::Cx => 16,
        Gate::Cz => 17,
        Gate::Swap => 18,
        Gate::Cu(_) => 19,
        Gate::Mcu(_) => 20,
        Gate::Fused(_) => 21,
        Gate::BatchPhase(_) => 22,
        Gate::BatchRzz(_) => 23,
        Gate::DiagonalBatch(_) => 24,
        Gate::MultiFused(_) => 25,
        Gate::Fused2q(_) => 26,
        Gate::Multi2q(_) => 27,
        Gate::QftBlock { .. } => 28,
        Gate::PauliRot(_) => 29,
        Gate::MultiPauliRot(_) => 30,
        Gate::Unitary(_) => 31,
    }
}

const GATE_VARIANTS: usize = 32;

/// One instance of every variant, in declaration order.
fn one_of_each_variant() -> Vec<Gate> {
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let h = Gate::H.matrix_2x2();
    let dft: Vec<Complex64> = (0..64)
        .map(|k| {
            let angle = std::f64::consts::TAU * ((k / 8) * (k % 8)) as f64 / 8.0;
            Complex64::from_polar(8f64.sqrt().recip(), angle)
        })
        .collect();
    let mut h_on_first = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (r, row) in h_on_first.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            if r % 2 == c % 2 {
                *value = h[r / 2][c / 2];
            }
        }
    }
    vec![
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
        Gate::Ry(0.3),
        Gate::Rz(0.3),
        Gate::P(0.3),
        Gate::Rzz(0.3),
        Gate::Cx,
        Gate::Cz,
        Gate::Swap,
        Gate::Cu(Box::new(h)),
        Gate::mcu(Gate::X.matrix_2x2(), 2),
        Gate::Fused(Box::new(h)),
        Gate::BatchPhase(Box::new(BatchPhaseData::new([(1, i), (2, -one)]))),
        Gate::BatchRzz(Box::new(BatchRzzData::new(vec![(0, 1, 0.3), (1, 2, 0.5)]))),
        Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(vec![
            DiagEntry::Phase1q {
                qubit: 0,
                d0: one,
                d1: i,
            },
            DiagEntry::Phase2q {
                q0: 1,
                q1: 2,
                phase: -one,
            },
        ]))),
        Gate::MultiFused(Box::new(MultiFusedData::new(vec![
            (0, h),
            (1, Gate::T.matrix_2x2()),
            (2, Gate::X.matrix_2x2()),
        ]))),
        Gate::Fused2q(Box::new(h_on_first)),
        Gate::Multi2q(Box::new(Multi2qData::new(vec![
            (0, 1, Gate::Cx.matrix_4x4()),
            (1, 2, Gate::Cz.matrix_4x4()),
        ]))),
        Gate::QftBlock {
            start: 0,
            num: 3,
            inverse: false,
            swaps: true,
            big_endian: false,
        },
        Gate::PauliRot(Box::new(PauliRotData {
            theta: 0.3,
            axes: vec![PauliAxis::X, PauliAxis::Z],
        })),
        Gate::MultiPauliRot(Box::new(MultiPauliRotData {
            rotations: vec![(0b011, 0, 0.3), (0, 0b110, 0.2)],
        })),
        Gate::Unitary(Box::new(UnitaryData {
            mat: dft,
            num_qubits: 3,
        })),
    ]
}

/// Instances whose classification depends on their data rather than their variant.
fn data_dependent_instances() -> Vec<Gate> {
    let x = Gate::X.matrix_2x2();
    let t = Gate::T.matrix_2x2();
    let h = Gate::H.matrix_2x2();
    vec![
        Gate::Cu(Box::new(t)),
        Gate::Cu(Box::new(x)),
        Gate::Fused(Box::new(t)),
        Gate::Fused(Box::new(x)),
        Gate::mcu(h, 2),
        Gate::PauliRot(Box::new(PauliRotData {
            theta: 0.3,
            axes: vec![PauliAxis::Z, PauliAxis::Z],
        })),
    ]
}

/// Whether every column of the row-major square `mat` holds exactly one nonzero entry.
fn is_monomial(mat: &[Complex64]) -> bool {
    let dim = mat.len().isqrt();
    (0..dim).all(|c| {
        (0..dim)
            .filter(|&r| mat[r * dim + c].norm() > 1e-12)
            .count()
            == 1
    })
}

#[test]
fn one_of_each_variant_covers_every_variant() {
    let indices: Vec<usize> = one_of_each_variant().iter().map(variant_index).collect();
    assert_eq!(indices, (0..GATE_VARIANTS).collect::<Vec<_>>());
}

#[test]
fn test_preserves_sparsity() {
    // A gate keeps the nonzero count iff it maps basis states to basis states, one
    // nonzero per column. The diagonal batches carry no dense matrix and are diagonal by
    // construction; the tiled variants report false whatever their factors.
    for gate in one_of_each_variant()
        .into_iter()
        .chain(data_dependent_instances())
    {
        let expected = match gate.dense_matrix() {
            Some(mat) => is_monomial(&mat),
            None => matches!(
                gate,
                Gate::BatchPhase(_) | Gate::BatchRzz(_) | Gate::DiagonalBatch(_)
            ),
        };
        assert_eq!(gate.preserves_sparsity(), expected, "{gate:?}");
    }
}

#[test]
fn dense_matrix_is_none_for_narrow_batched_and_tiled_variants() {
    let one = Complex64::new(1.0, 0.0);
    let h = Gate::H.matrix_2x2();
    let narrow = [
        Gate::BatchPhase(Box::new(BatchPhaseData::new([(1, -one)]))),
        Gate::BatchRzz(Box::new(BatchRzzData::new(vec![(0, 1, 0.3)]))),
        Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(vec![DiagEntry::Phase1q {
            qubit: 0,
            d0: one,
            d1: -one,
        }]))),
        Gate::MultiPauliRot(Box::new(MultiPauliRotData {
            rotations: vec![(0b11, 0, 0.3)],
        })),
        Gate::MultiFused(Box::new(MultiFusedData::new(vec![(0, h)]))),
        Gate::MultiFused(Box::new(MultiFusedData::new(vec![(0, h), (1, h)]))),
        Gate::Multi2q(Box::new(Multi2qData::new(vec![(
            0,
            1,
            Gate::Cx.matrix_4x4(),
        )]))),
        Gate::QftBlock {
            start: 0,
            num: 2,
            inverse: false,
            swaps: true,
            big_endian: false,
        },
    ];
    for gate in narrow {
        assert!(gate.num_qubits() <= 2, "{gate:?}");
        assert!(gate.dense_matrix().is_none(), "{gate:?}");
    }
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

    let h_mat = Gate::H.matrix_2x2();
    let cu_h = Gate::Cu(Box::new(h_mat));
    assert!(cu_h.controlled_phase().is_none());

    let z_mat = Gate::Z.matrix_2x2();
    let cu_z = Gate::Cu(Box::new(z_mat));
    assert!(cu_z.controlled_phase().is_some());
    let z_phase = cu_z.controlled_phase().unwrap();
    assert!((z_phase.re - (-1.0)).abs() < 1e-14);

    // Diagonal, but mat[0][0] != 1.
    let rz_mat = Gate::Rz(0.5).matrix_2x2();
    let cu_rz = Gate::Cu(Box::new(rz_mat));
    assert!(cu_rz.controlled_phase().is_none());

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
    for gate in one_of_each_variant()
        .into_iter()
        .chain(data_dependent_instances())
    {
        let expected = gate.num_qubits() == 1 && {
            let m = gate.matrix_2x2();
            m[0][1].norm() < 1e-12 && m[1][0].norm() < 1e-12
        };
        assert_eq!(gate.is_diagonal_1q(), expected, "{gate:?}");
    }
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

// The tolerance is a per-entry magnitude, so the boundary sits where the
// entries differ by 1e-12 and not at its square root. Rx(1e-13) has
// off-diagonal entries of 5e-14, Rx(1e-11) has 5e-12.
#[test]
fn test_recognize_tolerance_is_a_per_entry_magnitude() {
    let inside = Gate::Rx(1e-13).matrix_2x2();
    assert_eq!(Gate::recognize_matrix(&inside), Some(Gate::Id));

    let outside = Gate::Rx(1e-11).matrix_2x2();
    assert_eq!(Gate::recognize_matrix(&outside), None);
}

// A rotation small enough to read as identity under a squared tolerance still
// moves amplitude, and a phase a hair off pi/2 is not S.
#[test]
fn test_recognize_rejects_a_gate_that_still_moves_amplitude() {
    let rx = Gate::Rx(1e-5).matrix_2x2();
    assert_eq!(Gate::recognize_matrix(&rx), None);

    let near_s = Gate::P(std::f64::consts::FRAC_PI_2 + 1e-6).matrix_2x2();
    assert_eq!(Gate::recognize_matrix(&near_s), None);
    assert!(matches!(
        Gate::recognize_matrix_up_to_phase(&near_s),
        Some((Gate::Rz(_), _))
    ));
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

#[test]
fn test_recognize_up_to_phase_returns_the_scalar() {
    let phase = Complex64::from_polar(1.0, 0.42);
    for gate in [
        Gate::H,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::S,
        Gate::T,
        Gate::SX,
    ] {
        let m = gate.matrix_2x2();
        let phased = [
            [m[0][0] * phase, m[0][1] * phase],
            [m[1][0] * phase, m[1][1] * phase],
        ];
        let (named, scalar) = Gate::recognize_matrix_up_to_phase(&phased).unwrap();
        assert_eq!(named, gate);
        assert!((scalar - phase).norm() < 1e-9, "{gate:?}: {scalar}");
    }

    let (rz, scalar) = Gate::recognize_matrix_up_to_phase(&Gate::P(0.3).matrix_2x2()).unwrap();
    assert!(matches!(rz, Gate::Rz(theta) if (theta - 0.3).abs() < 1e-12));
    assert!((scalar - Complex64::from_polar(1.0, 0.15)).norm() < 1e-12);

    assert_eq!(
        Gate::recognize_matrix_up_to_phase(&Gate::Rx(0.7).matrix_2x2()),
        None
    );
}

#[test]
#[should_panic(expected = "at least one control qubit")]
fn mcu_with_zero_controls_panics() {
    let _ = Gate::mcu(Gate::X.matrix_2x2(), 0);
}

#[test]
#[should_panic(expected = "McuData needs at least 2 controls")]
fn mcu_data_with_one_control_panics() {
    let _ = McuData::new(Gate::X.matrix_2x2(), 1);
}

#[test]
#[should_panic(expected = "BatchPhaseData holds at most 40 entries")]
fn batch_phase_data_past_its_capacity_panics() {
    let one = Complex64::new(1.0, 0.0);
    let _ = BatchPhaseData::new((1..=BatchPhaseData::MAX_PHASES + 1).map(|q| (q, one)));
}

#[test]
#[should_panic(expected = "BatchRzzData holds at most 32 edges")]
fn batch_rzz_data_past_its_capacity_panics() {
    let edges = (0..=BatchRzzData::MAX_EDGES)
        .map(|q| (q, q + 1, 0.1))
        .collect();
    let _ = BatchRzzData::new(edges);
}

#[test]
#[should_panic(expected = "matrix_2x2 called on 2-qubit gate `cx`")]
fn matrix_2x2_on_a_two_qubit_gate_panics() {
    let _ = Gate::Cx.matrix_2x2();
}

#[test]
#[should_panic(expected = "matrix_2x2 called on 2-qubit gate")]
fn matrix_2x2_on_a_weight_two_pauli_rotation_panics() {
    let _ = Gate::PauliRot(Box::new(PauliRotData {
        theta: 0.3,
        axes: vec![PauliAxis::X, PauliAxis::Z],
    }))
    .matrix_2x2();
}

#[test]
#[should_panic(expected = "matrix_4x4 called on non-standard-2q gate `h`")]
fn matrix_4x4_on_a_single_qubit_gate_panics() {
    let _ = Gate::H.matrix_4x4();
}

#[test]
#[should_panic(expected = "matrix_4x4 called on non-standard-2q gate")]
fn matrix_4x4_on_a_weight_three_pauli_rotation_panics() {
    let _ = Gate::PauliRot(Box::new(PauliRotData {
        theta: 0.3,
        axes: vec![PauliAxis::X, PauliAxis::Y, PauliAxis::Z],
    }))
    .matrix_4x4();
}

// The check is a `debug_assert`, so the release-profile lane skips this case.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "matrix_power_real only for single-qubit gates")]
fn matrix_power_real_on_a_two_qubit_gate_panics() {
    let _ = Gate::Cx.matrix_power_real(0.5);
}

/// Flatten a nested matrix into the row-major form `Gate::unitary` takes.
fn flat<const N: usize>(rows: &[[Complex64; N]; N]) -> Vec<Complex64> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

/// `mat` lifted to `k` qubits as the identity outside its trailing 2x2 block.
fn controlled(mat: &[[Complex64; 2]; 2], k: usize) -> Vec<Complex64> {
    let dim = 1usize << k;
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim - 2 {
        out[i * dim + i] = Complex64::new(1.0, 0.0);
    }
    for row in 0..2 {
        for column in 0..2 {
            out[(dim - 2 + row) * dim + (dim - 2 + column)] = mat[row][column];
        }
    }
    out
}

/// The `2^k`-point DFT matrix: dense, complex, and unitary, so no detector in
/// the constructor can lower it.
fn dft(k: usize) -> Vec<Complex64> {
    let dim = 1usize << k;
    let scale = 1.0 / (dim as f64).sqrt();
    let mut out = Vec::with_capacity(dim * dim);
    for row in 0..dim {
        for column in 0..dim {
            let angle = std::f64::consts::TAU * (row * column % dim) as f64 / dim as f64;
            out.push(Complex64::from_polar(scale, angle));
        }
    }
    out
}

#[test]
fn unitary_recognizes_a_one_qubit_named_gate() {
    let gate = Gate::unitary(flat(&Gate::H.matrix_2x2()), 1).unwrap();
    assert_eq!(gate, Gate::H);
    assert!(gate.is_clifford());
}

#[test]
fn unitary_keeps_an_unnamed_one_qubit_matrix_as_a_fused_matrix() {
    let gate = Gate::unitary(flat(&Gate::Rx(0.7).matrix_2x2()), 1).unwrap();
    assert!(matches!(gate, Gate::Fused(_)), "{gate:?}");
    assert_eq!(gate.matrix_2x2(), Gate::Rx(0.7).matrix_2x2());
}

#[test]
fn unitary_recognizes_the_named_two_qubit_gates() {
    for named in [Gate::Cx, Gate::Cz, Gate::Swap] {
        assert_eq!(Gate::unitary(flat(&named.matrix_4x4()), 2).unwrap(), named);
    }
    assert!(matches!(
        Gate::unitary(flat(&Gate::Rzz(0.4).matrix_4x4()), 2).unwrap(),
        Gate::Rzz(theta) if (theta - 0.4).abs() < 1e-12
    ));
    assert!(matches!(
        Gate::unitary(flat(&Gate::cu(Gate::H.matrix_2x2()).matrix_4x4()), 2).unwrap(),
        Gate::Cu(_)
    ));
}

#[test]
fn unitary_lowers_an_identity_blocked_matrix_to_mcu() {
    for k in 3..=MAX_UNITARY_QUBITS {
        let gate = Gate::unitary(controlled(&Gate::X.matrix_2x2(), k), k).unwrap();
        let Gate::Mcu(data) = &gate else {
            panic!("expected Mcu at k = {k}, got {gate:?}");
        };
        assert_eq!(data.num_controls as usize, k - 1);
        assert_eq!(gate.num_qubits(), k);
    }
}

#[test]
fn unitary_keeps_a_dense_wide_matrix() {
    for k in 3..=MAX_UNITARY_QUBITS {
        let gate = Gate::unitary(dft(k), k).unwrap();
        assert!(matches!(gate, Gate::Unitary(_)), "{gate:?}");
        assert_eq!(gate.num_qubits(), k);
        assert_eq!(gate.name(), "unitary");
    }
}

#[test]
fn unitary_rejects_a_matrix_that_is_not_unitary() {
    let mut mat = flat(&Gate::Cx.matrix_4x4());
    mat[0] = Complex64::new(0.5, 0.0);
    let err = Gate::unitary(mat, 2).unwrap_err().to_string();
    assert!(err.contains("not unitary"), "{err}");

    let err = Gate::unitary(vec![Complex64::new(1.0, 0.0); 5], 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("4 entries"), "{err}");

    let err = Gate::unitary(vec![Complex64::new(f64::NAN, 0.0); 4], 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not finite"), "{err}");

    let err = Gate::unitary(vec![Complex64::new(0.0, 0.0); 1 << 12], 6)
        .unwrap_err()
        .to_string();
    assert!(err.contains("between 1 and"), "{err}");
}

#[test]
fn unitary_inverse_is_the_adjoint() {
    let mat = dft(3);
    let Gate::Unitary(inverse) = Gate::unitary(mat.clone(), 3).unwrap().inverse() else {
        panic!("expected a dense inverse");
    };
    for row in 0..8 {
        for column in 0..8 {
            let want = mat[column * 8 + row].conj();
            assert!((inverse.matrix()[row * 8 + column] - want).norm() < 1e-15);
        }
    }
}

#[test]
fn mcu_with_one_control_is_a_cu() {
    let mat = Gate::H.matrix_2x2();
    assert_eq!(Gate::mcu(mat, 1), Gate::cu(mat));
    assert!(matches!(Gate::mcu(mat, 2), Gate::Mcu(_)));
}

#[test]
fn a_weight_one_pauli_rotation_has_the_axis_rotation_matrix() {
    for (axis, named) in [
        (PauliAxis::X, Gate::Rx(0.7)),
        (PauliAxis::Y, Gate::Ry(0.7)),
        (PauliAxis::Z, Gate::Rz(0.7)),
    ] {
        let rot = Gate::PauliRot(Box::new(PauliRotData {
            theta: 0.7,
            axes: vec![axis],
        }));
        assert_eq!(rot.matrix_2x2(), named.matrix_2x2());
    }
}

#[test]
fn multi_fused_data_computes_its_diagonal_flag() {
    let h = Gate::H.matrix_2x2();
    let t = Gate::T.matrix_2x2();
    assert!(MultiFusedData::new(vec![(0, t), (1, Gate::Z.matrix_2x2())]).all_diagonal());
    assert!(!MultiFusedData::new(vec![(0, t), (1, h)]).all_diagonal());
    assert_eq!(MultiFusedData::new(vec![(2, h)]).gates(), &[(2, h)]);
}

#[test]
fn multi_fused_data_folds_repeated_targets_in_order() {
    let h = Gate::H.matrix_2x2();
    let t = Gate::T.matrix_2x2();
    let x = Gate::X.matrix_2x2();
    let data = MultiFusedData::new(vec![(0, h), (1, x), (0, t)]);
    assert_eq!(data.gates(), &[(0, mat_mul_2x2(&t, &h)), (1, x)]);
    assert!(!data.all_diagonal());
    assert!(MultiFusedData::new(vec![(3, h), (3, h)]).all_diagonal());
}
