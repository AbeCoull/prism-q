use super::*;
use crate::circuit::Circuit;
use crate::sim;

const EPS: f64 = 1e-10;

fn run_mps(circuit: &Circuit) -> MpsBackend {
    let mut b = MpsBackend::new(42, 64);
    sim::run_on(&mut b, circuit).unwrap();
    b
}

fn run_mps_probs(circuit: &Circuit) -> Vec<f64> {
    let b = run_mps(circuit);
    b.probabilities().unwrap()
}

fn assert_probs_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < EPS, "prob[{i}]: expected {e}, got {a}");
    }
}

fn statevector_pauli_expectation(
    circuit: &Circuit,
    pauli_factors: &[(usize, MpsPauliAxis)],
) -> Complex64 {
    // Build dense amplitudes and contract the Pauli expectation directly.
    let n = circuit.num_qubits;
    let mut backend = crate::backend::statevector::StatevectorBackend::new(42);
    crate::backend::Backend::init(&mut backend, n, 0).unwrap();
    crate::backend::Backend::apply_instructions(&mut backend, &circuit.instructions).unwrap();
    let amps = crate::backend::Backend::export_statevector(&backend).unwrap();

    // ⟨ψ|P|ψ⟩ = Σ_{x,y} ψ*_x P_{x,y} ψ_y
    // For a Pauli string P = ⊗ P_i, the matrix element is non-zero
    // only when y differs from x by the X-bits of P, and the value
    // is (-1)^(z_bits·x) · i^(num_y_factors).
    let mut x_mask = 0usize;
    let mut z_mask = 0usize;
    let mut num_y = 0usize;
    for &(q, axis) in pauli_factors {
        match axis {
            MpsPauliAxis::X => x_mask |= 1 << q,
            MpsPauliAxis::Z => z_mask |= 1 << q,
            MpsPauliAxis::Y => {
                x_mask |= 1 << q;
                z_mask |= 1 << q;
                num_y += 1;
            }
        }
    }
    let i_factor = match num_y % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    let mut sum = Complex64::new(0.0, 0.0);
    for x in 0..(1 << n) {
        let y = x ^ x_mask;
        let sign = if (z_mask & x).count_ones() & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        sum += amps[x].conj() * (Complex64::new(sign, 0.0) * i_factor) * amps[y];
    }
    sum
}

// The sampler picks each site from `site_conditional_weights`, so the
// product of the conditionals along a path is the probability it draws
// that path with. Comparing it to the dense vector pins the sampled
// distribution exactly rather than statistically, and covers the
// site-to-logical mapping the SWAP-routed layout leaves behind.
#[test]
fn mps_conditional_path_probabilities_match_the_dense_vector() {
    let mut c = Circuit::new(5, 0);
    for q in 0..5 {
        c.add_gate(Gate::Ry(0.4 + 0.2 * q as f64), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_gate(Gate::Cx, &[4, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[2, 0]);
    let b = run_mps(&c);

    let dense = b.probabilities().unwrap();
    let right = b.right_environments();
    let max_bond = b
        .sites
        .iter()
        .map(|site| site.bond_left.max(site.bond_right))
        .max()
        .unwrap();

    for (basis, &expected) in dense.iter().enumerate() {
        assert!(
            expected > 1e-6,
            "basis {basis} carries probability {expected:.3e}; the case is meant to have \
             full support so every conditional is exercised"
        );

        let mut left = vec![ZERO; max_bond];
        let mut w = vec![ZERO; 2 * max_bond];
        left[0] = ONE;
        let mut joint = 1.0f64;
        for (site, right_env) in right.iter().enumerate() {
            let br = b.sites[site].bond_right;
            let prob = b.site_conditional_weights(site, &left, right_env, &mut w);
            let bit = (basis >> b.logical_for_site(site)) & 1;
            joint *= prob[bit] / (prob[0] + prob[1]);

            let scale = 1.0 / prob[bit].sqrt();
            left[..br].copy_from_slice(&w[bit * br..(bit + 1) * br]);
            for value in &mut left[..br] {
                *value *= scale;
            }
            left[br..].fill(ZERO);
        }
        assert!(
            (joint - expected).abs() < 1e-12,
            "basis {basis}: conditional path gives {joint}, dense vector gives {expected}"
        );
    }
}

#[test]
fn mps_pauli_expectation_z_string_matches_statevector_on_h_t_circuit() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[1, 2]);
    let mps = run_mps(&c);

    for factors in [
        vec![(0usize, MpsPauliAxis::Z)],
        vec![(1, MpsPauliAxis::Z)],
        vec![(2, MpsPauliAxis::Z)],
        vec![(0, MpsPauliAxis::Z), (1, MpsPauliAxis::Z)],
        vec![(0, MpsPauliAxis::Z), (2, MpsPauliAxis::Z)],
        vec![
            (0, MpsPauliAxis::Z),
            (1, MpsPauliAxis::Z),
            (2, MpsPauliAxis::Z),
        ],
    ] {
        let mps_val = mps.pauli_expectation(&factors).unwrap();
        let sv_val = statevector_pauli_expectation(&c, &factors);
        assert!(
            (mps_val - sv_val).norm() < 1e-8,
            "factors={factors:?}: mps={mps_val:?}, sv={sv_val:?}"
        );
    }
}

#[test]
fn mps_pauli_expectation_mixed_xyz_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mps = run_mps(&c);

    for factors in [
        vec![(0usize, MpsPauliAxis::X)],
        vec![(0, MpsPauliAxis::Y)],
        vec![(1, MpsPauliAxis::X), (0, MpsPauliAxis::Z)],
        vec![(0, MpsPauliAxis::Y), (1, MpsPauliAxis::Y)],
    ] {
        let mps_val = mps.pauli_expectation(&factors).unwrap();
        let sv_val = statevector_pauli_expectation(&c, &factors);
        assert!(
            (mps_val - sv_val).norm() < 1e-8,
            "factors={factors:?}: mps={mps_val:?}, sv={sv_val:?}"
        );
    }
}

#[test]
fn mps_pauli_expectation_returns_one_for_normalized_state() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mps = run_mps(&c);
    let val = mps.pauli_expectation(&[]).unwrap();
    assert!(
        (val - Complex64::new(1.0, 0.0)).norm() < 1e-10,
        "⟨ψ|ψ⟩ = {val:?}, expected 1"
    );
}

#[test]
fn test_svd_2x2() {
    let a = vec![
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let r = svd(&a, 2, 2);
    assert_eq!(r.s.len(), 2);
    assert!(r.s[0] >= r.s[1]);

    let mut recon = [ZERO; 4];
    for c in 0..2 {
        for row in 0..2 {
            for kk in 0..2 {
                recon[c * 2 + row] += r.u[kk * r.u_rows + row]
                    * Complex64::new(r.s[kk], 0.0)
                    * r.vt[kk * r.vt_cols + c];
            }
        }
    }
    for i in 0..4 {
        assert!(
            (recon[i] - a[i]).norm() < 1e-10,
            "recon[{i}] = {:?}, expected {:?}",
            recon[i],
            a[i]
        );
    }
}

#[test]
fn test_svd_rank_deficient() {
    let a = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let r = svd(&a, 2, 2);
    assert!(r.s[1] < 1e-10, "second singular value should be ~0");
}

#[test]
fn test_svd_identity() {
    let a = vec![ONE, ZERO, ZERO, ONE];
    let r = svd(&a, 2, 2);
    assert!((r.s[0] - 1.0).abs() < 1e-10);
    assert!((r.s[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_svd_wide_matrix() {
    let a = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 1.0),
    ];
    let r = svd(&a, 2, 3);
    assert_eq!(r.u_rows, 2);
    assert_eq!(r.vt_cols, 3);

    let mut recon = [ZERO; 6];
    for c in 0..3 {
        for row in 0..2 {
            for kk in 0..2 {
                recon[c * 2 + row] += r.u[kk * r.u_rows + row]
                    * Complex64::new(r.s[kk], 0.0)
                    * r.vt[kk * r.vt_cols + c];
            }
        }
    }
    for i in 0..6 {
        assert!(
            (recon[i] - a[i]).norm() < 1e-10,
            "recon[{i}] = {:?}, expected {:?}",
            recon[i],
            a[i]
        );
    }
}

#[test]
fn test_init_zero_state() {
    let mut b = MpsBackend::new(42, 64);
    b.init(3, 0).unwrap();
    assert_eq!(b.sites.len(), 3);
    for s in &b.sites {
        assert_eq!(s.bond_left, 1);
        assert_eq!(s.bond_right, 1);
        assert_eq!(s.data.len(), 2);
        assert!((s.data[0] - ONE).norm() < EPS);
        assert!((s.data[1] - ZERO).norm() < EPS);
    }
}

#[test]
fn test_x_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    assert_probs_close(&run_mps_probs(&c), &[0.0, 1.0]);
}

#[test]
fn test_h_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    assert_probs_close(&run_mps_probs(&c), &[0.5, 0.5]);
}

#[test]
fn test_hh_is_identity() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[0]);
    assert_probs_close(&run_mps_probs(&c), &[1.0, 0.0]);
}

#[test]
fn test_rz_preserves_zero() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rz(1.234), &[0]);
    assert_probs_close(&run_mps_probs(&c), &[1.0, 0.0]);
}

#[test]
fn test_rx_pi() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(std::f64::consts::PI), &[0]);
    assert_probs_close(&run_mps_probs(&c), &[0.0, 1.0]);
}

#[test]
fn test_bell_state() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs_close(&run_mps_probs(&c), &[0.5, 0.0, 0.0, 0.5]);
}

#[test]
fn test_bell_bond_dim() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let b = run_mps(&c);
    assert_eq!(b.sites[0].bond_right, 2);
    assert_eq!(b.sites[1].bond_left, 2);
}

#[test]
fn test_cx_no_flip() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_probs_close(&run_mps_probs(&c), &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_cz_phase() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    assert_probs_close(&run_mps_probs(&c), &[0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_swap() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    assert_probs_close(&run_mps_probs(&c), &[0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn test_ghz_3() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    let probs = run_mps_probs(&c);
    assert_probs_close(&probs, &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]);
}

#[test]
fn test_non_adjacent_cx() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Cx, &[0, 2]);
    assert_probs_close(
        &run_mps_probs(&c),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
}

#[test]
fn test_measure_deterministic() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let b = run_mps(&c);
    assert!(b.classical_results()[0]);
}

#[test]
fn test_measure_seeded() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    let b1 = run_mps(&c);
    let b2 = run_mps(&c);
    assert_eq!(b1.classical_results()[0], b2.classical_results()[0]);
}

#[test]
fn test_fused_gate() {
    let h_mat = Gate::H.matrix_2x2();
    let t_mat = Gate::T.matrix_2x2();
    let mut fused = [[ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                fused[i][j] += t_mat[i][k] * h_mat[k][j];
            }
        }
    }

    let mut c1 = Circuit::new(1, 0);
    c1.add_gate(Gate::H, &[0]);
    c1.add_gate(Gate::T, &[0]);
    let p1 = run_mps_probs(&c1);

    let mut c2 = Circuit::new(1, 0);
    c2.add_gate(Gate::Fused(Box::new(fused)), &[0]);
    let p2 = run_mps_probs(&c2);

    assert_probs_close(&p1, &p2);
}

#[test]
fn test_supports_fused_gates() {
    let b = MpsBackend::new(42, 64);
    assert!(b.supports_fused_gates());
}

#[test]
fn test_probabilities_cap() {
    let mut b = MpsBackend::new(42, 64);
    b.init(usize::BITS as usize, 0).unwrap();
    assert!(b.probabilities().is_err());
}

#[test]
fn test_mcu_matrix_toffoli() {
    let x_mat = Gate::X.matrix_2x2();
    let order = vec![0, 1, 2]; // ctrl0, ctrl1, target, identity order
    let gate = mcu_matrix(2, &x_mat, &order);
    // 8×8 matrix: identity for states 0..5, then X on target for states 6,7
    // state 6 = |110⟩, state 7 = |111⟩ → swap these
    assert!((gate[6 * 8 + 6] - ZERO).norm() < 1e-12); // 6→6 should be 0
    assert!((gate[7 * 8 + 6] - ONE).norm() < 1e-12); // 6→7
    assert!((gate[6 * 8 + 7] - ONE).norm() < 1e-12); // 7→6
    assert!((gate[7 * 8 + 7] - ZERO).norm() < 1e-12); // 7→7 should be 0
    // Diagonal entries for 0..5 should be 1
    for s in 0..6 {
        assert!((gate[s * 8 + s] - ONE).norm() < 1e-12, "state {s}");
    }
}

fn assert_mps_matches_statevector(circuit: &crate::circuit::Circuit) {
    use crate::backend::statevector::StatevectorBackend;

    let mut sv = StatevectorBackend::new(42);
    sv.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for inst in &circuit.instructions {
        sv.apply(inst).unwrap();
    }
    let sv_probs = sv.probabilities().unwrap();

    let mut mps = MpsBackend::new(42, 128);
    mps.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for inst in &circuit.instructions {
        mps.apply(inst).unwrap();
    }
    let mps_probs = mps.probabilities().unwrap();

    for (i, (a, b)) in sv_probs.iter().zip(&mps_probs).enumerate() {
        assert!((a - b).abs() < 1e-10, "prob[{i}]: sv={a}, mps={b}");
    }
}

#[test]
fn test_toffoli_adjacent() {
    use crate::circuit::Circuit;
    use crate::gates::McuData;

    let x_mat = Gate::X.matrix_2x2();
    let mut c = Circuit::new(3, 0);
    // Set controls to |1⟩
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    // Toffoli: should flip target
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: x_mat,
            num_controls: 2,
        })),
        &[0, 1, 2],
    );
    assert_mps_matches_statevector(&c);
}

#[test]
fn test_toffoli_no_flip() {
    use crate::circuit::Circuit;
    use crate::gates::McuData;

    let x_mat = Gate::X.matrix_2x2();
    let mut c = Circuit::new(3, 0);
    // Only one control is set, should NOT flip target
    c.add_gate(Gate::X, &[0]);
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: x_mat,
            num_controls: 2,
        })),
        &[0, 1, 2],
    );
    assert_mps_matches_statevector(&c);
}

#[test]
fn test_toffoli_non_adjacent() {
    use crate::circuit::Circuit;
    use crate::gates::McuData;

    let x_mat = Gate::X.matrix_2x2();
    let mut c = Circuit::new(5, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: x_mat,
            num_controls: 2,
        })),
        &[0, 2, 4],
    );
    assert_mps_matches_statevector(&c);
}

#[test]
fn test_cccx() {
    use crate::circuit::Circuit;
    use crate::gates::McuData;

    let x_mat = Gate::X.matrix_2x2();
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: x_mat,
            num_controls: 3,
        })),
        &[0, 1, 2, 3],
    );
    assert_mps_matches_statevector(&c);
}

#[test]
fn test_mcu_arbitrary_unitary() {
    use crate::circuit::Circuit;
    use crate::gates::McuData;

    let ry_mat = Gate::Ry(std::f64::consts::FRAC_PI_4).matrix_2x2();
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: ry_mat,
            num_controls: 2,
        })),
        &[0, 1, 2],
    );
    assert_mps_matches_statevector(&c);
}

#[test]
fn test_non_adjacent_layout_tracks_logical_targets() {
    let mut c = Circuit::new(6, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[5]);
    c.add_gate(Gate::Cx, &[0, 5]);
    c.add_gate(Gate::Ry(0.37), &[0]);
    c.add_gate(Gate::Rz(-0.52), &[5]);
    c.add_gate(Gate::Swap, &[0, 3]);
    c.add_gate(Gate::S, &[3]);
    c.add_gate(Gate::Cx, &[1, 4]);
    c.add_gate(Gate::H, &[4]);
    assert_mps_matches_statevector(&c);
}

#[test]
fn canonicalize_logical_order_preserves_state() {
    let mut c = Circuit::new(6, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[5]);
    c.add_gate(Gate::Cx, &[0, 5]);
    c.add_gate(Gate::Ry(0.37), &[2]);
    c.add_gate(Gate::Cz, &[2, 5]);

    let mut b = run_mps(&c);
    let before = b.export_statevector().unwrap();
    b.canonicalize_logical_order().unwrap();
    assert_eq!(b.logical_to_site, vec![0, 1, 2, 3, 4, 5]);
    let after = b.export_statevector().unwrap();
    for (i, (a, e)) in after.iter().zip(&before).enumerate() {
        assert!(
            (*a - *e).norm() < EPS,
            "amp[{i}] differs: actual={a:?}, expected={e:?}"
        );
    }
}

#[test]
fn test_measure_after_non_adjacent_routing_uses_logical_qubit() {
    let mut c = Circuit::new(5, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Cx, &[0, 4]);
    c.add_measure(4, 0);
    assert_mps_matches_statevector(&c);

    let b = run_mps(&c);
    assert_eq!(b.classical_results(), &[true]);
}

#[test]
fn test_reset_after_non_adjacent_routing_uses_logical_qubit() {
    let mut c = Circuit::new(5, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Cx, &[0, 4]);
    c.add_reset(0);
    c.add_gate(Gate::H, &[4]);
    assert_mps_matches_statevector(&c);
}

#[test]
fn is_qubit_in_zero_state_basic() {
    use crate::circuit::Circuit;

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::X, &[1]);
    let b = run_mps(&c);
    assert!(b.is_qubit_in_zero_state(0, 1e-10).unwrap());
    assert!(!b.is_qubit_in_zero_state(1, 1e-10).unwrap());
    assert!(b.is_qubit_in_zero_state(2, 1e-10).unwrap());
}

#[test]
fn is_qubit_in_zero_state_superposition_not_zero() {
    use crate::circuit::Circuit;

    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    let b = run_mps(&c);
    assert!(!b.is_qubit_in_zero_state(0, 1e-10).unwrap());
    assert!(b.is_qubit_in_zero_state(1, 1e-10).unwrap());
}

#[test]
fn is_qubit_in_zero_state_entangled_marginal_nonzero() {
    use crate::circuit::Circuit;

    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let b = run_mps(&c);
    assert!(!b.is_qubit_in_zero_state(0, 1e-10).unwrap());
    assert!(!b.is_qubit_in_zero_state(1, 1e-10).unwrap());
}

#[test]
fn test_batch_phase_decomposition() {
    use crate::circuit::Circuit;
    use crate::gates::BatchPhaseData;

    let phase1 = Complex64::from_polar(1.0, 0.5);
    let phase2 = Complex64::from_polar(1.0, 1.2);

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(
        Gate::BatchPhase(Box::new(BatchPhaseData {
            phases: smallvec::smallvec![(1, phase1), (2, phase2)],
        })),
        &[0, 1, 2],
    );
    assert_mps_matches_statevector(&c);
}

#[test]
fn svd_epsilon_default_is_pinned() {
    let b = MpsBackend::new(42, 64);
    assert_eq!(b.svd_epsilon, 1e-12);
}

#[test]
#[should_panic(expected = "svd epsilon")]
fn svd_epsilon_rejects_one() {
    MpsBackend::new(42, 64).set_svd_epsilon(1.0);
}

// Uncapped brickwork saturates its width ceiling, so a raised threshold must
// show as a lower peak bond, a reported discard, and a realized error within
// a small factor of that discard (the estimate is first order, not a
// certificate, and the factor it understates by grows with the number of
// truncating SVDs; 10x is headroom, not a measured ratio).
#[test]
fn raised_epsilon_lowers_bond_and_reports_the_discard() {
    let circuit = crate::circuits::brickwork_circuit(14, 20, 42);

    let mut exact = MpsBackend::new(42, 4096);
    exact.init(14, 0).unwrap();
    exact.apply_instructions(&circuit.instructions).unwrap();
    let reference = exact.export_statevector().unwrap();
    let exact_bond = exact.current_max_bond_dim();

    let mut b = MpsBackend::new(42, 4096);
    b.set_svd_epsilon(1e-3);
    b.init(14, 0).unwrap();
    b.apply_instructions(&circuit.instructions).unwrap();

    assert!(
        b.current_max_bond_dim() < exact_bond,
        "raised threshold left the peak bond at {} against {exact_bond}",
        b.current_max_bond_dim()
    );
    let discarded = b.truncation_discarded();
    assert!(discarded > 0.0, "raised threshold reported no discard");
    match b.exactness() {
        crate::sim::Exactness::Approximate {
            fidelity_lower_bound: Some(bound),
        } => assert!((bound - (1.0 - discarded)).abs() < 1e-15),
        other => panic!("expected a reported bound, got {other:?}"),
    }

    let v = b.export_statevector().unwrap();
    let inner: Complex64 = reference.iter().zip(&v).map(|(r, x)| r.conj() * x).sum();
    let realized_err = 1.0 - inner.norm_sqr();
    assert!(
        realized_err < 10.0 * discarded,
        "realized error {realized_err:.3e} against reported discard {discarded:.3e}"
    );
}

// One scratch pair reused across overlaps of different bond shapes must give
// the same values as fresh-allocation calls; stale contents from a larger
// pair must not leak into a smaller one.
#[test]
fn inner_product_scratch_reuse_matches_fresh() {
    let run = |depth: usize, seed: u64| {
        let circuit = crate::circuits::brickwork_circuit(8, depth, seed);
        let mut b = MpsBackend::new(42, 64);
        b.init(8, 0).unwrap();
        b.apply_instructions(&circuit.instructions).unwrap();
        b
    };
    let big = run(8, 42);
    let small = run(2, 43);

    let mut tmp = Vec::new();
    let mut next_env = Vec::new();
    for (bra, ket) in [(&big, &small), (&small, &small), (&big, &big)] {
        let reused = bra
            .inner_product_with_scratch(ket, &mut tmp, &mut next_env)
            .unwrap();
        let fresh = bra.inner_product(ket).unwrap();
        assert_eq!(reused, fresh);
    }
}

// The cap ladder, which the epsilon test above does not reach: that one holds
// cap 4096 and varies the SVD threshold instead. The export is normalized, so
// the overlap reads direction error alone, and lost norm is read off the chain
// itself; an unnormalized overlap would count the same discarded weight twice,
// once as lost norm and once as direction error. Both
// halves stay inside the reported discard, at 0.69 and 0.72 of it by cap 64,
// and both go vacuous at cap 4, whose discard has passed 1. Monotonicity holds
// for this fixture rather than by construction: no canonical gauge is kept, so
// the subspace one cap keeps does not nest inside the next.
#[test]
fn tighter_caps_lose_more_and_report_it() {
    let circuit = crate::circuits::brickwork_circuit(14, 20, 42);

    let mut exact = MpsBackend::new(42, 4096);
    exact.init(14, 0).unwrap();
    exact.apply_instructions(&circuit.instructions).unwrap();
    let reference = exact.export_statevector().unwrap();
    assert!(exact.truncation_discarded() < 1e-20);

    // 2^7 bounds the Schmidt rank of any 14-qubit chain, so the 256 default
    // cannot truncate at this width and the ladder below is the coverage.
    let exact_bond = exact.current_max_bond_dim();
    assert!(exact_bond > 64, "caps under {exact_bond} have to truncate");

    let mut tighter_error = f64::INFINITY;
    for cap in [4usize, 16, 64] {
        let mut b = MpsBackend::new(42, cap);
        b.init(14, 0).unwrap();
        b.apply_instructions(&circuit.instructions).unwrap();

        let v = b.export_statevector().unwrap();
        let kept = b.pauli_expectation(&[]).unwrap().re;
        let inner: Complex64 = reference.iter().zip(&v).map(|(r, x)| r.conj() * x).sum();
        let realized = 1.0 - inner.norm_sqr();
        let discarded = b.truncation_discarded();

        assert!(
            realized <= tighter_error,
            "cap {cap} realized {realized:.3e}, no better than {tighter_error:.3e} at the cap below it"
        );
        tighter_error = realized;

        assert!(
            realized < 1.5 * discarded,
            "cap {cap} realized {realized:.3e} against reported discard {discarded:.3e}"
        );
        assert!(
            1.0 - kept < 1.5 * discarded,
            "cap {cap} kept only {kept:.3e} of the weight against reported discard {discarded:.3e}"
        );
    }
}

fn mps_after(circuit: &Circuit, cap: usize) -> MpsBackend {
    let mut b = MpsBackend::new(42, cap);
    b.init(circuit.num_qubits, 0).unwrap();
    b.apply_instructions(&circuit.instructions).unwrap();
    b
}

fn bell_pairs(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

fn interior_bonds(b: &MpsBackend) -> Vec<usize> {
    b.sites[..b.sites.len() - 1]
        .iter()
        .map(|t| t.bond_right)
        .collect()
}

// A sweep to the far end is a canonicalization: it factorizes every site it
// crosses whatever gauge that site was in. The shapes cover a saturated chain,
// an odd width, a truncated chain, interior bonds of 1, and a product state.
#[test]
fn move_center_makes_every_other_site_an_isometry() {
    for (label, circuit, cap, bond_range) in [
        (
            "brickwork_8",
            crate::circuits::brickwork_circuit(8, 6, 42),
            4096,
            (2, 8),
        ),
        (
            "brickwork_7",
            crate::circuits::brickwork_circuit(7, 6, 43),
            4096,
            (2, 8),
        ),
        (
            "brickwork_8_cap4",
            crate::circuits::brickwork_circuit(8, 6, 42),
            4,
            (2, 4),
        ),
        ("bell_pairs_6", bell_pairs(6), 4096, (1, 2)),
        ("product_5", Circuit::new(5, 0), 4096, (1, 1)),
    ] {
        let n = circuit.num_qubits;
        let mut b = mps_after(&circuit, cap);
        b.move_center(n - 1);
        b.assert_gauge(n - 1);

        let bonds = interior_bonds(&b);
        assert_eq!(
            (*bonds.iter().min().unwrap(), *bonds.iter().max().unwrap()),
            bond_range,
            "{label} bond profile {bonds:?}"
        );

        for target in (0..n).rev() {
            b.move_center(target);
            b.assert_gauge(target);
        }
        for target in 0..n {
            b.move_center(target);
            b.assert_gauge(target);
        }
    }
}

// Every ordered pair of positions, so a move that loses a singular value or
// mismatches a reshape shows up as a changed amplitude or a changed norm.
#[test]
fn moving_the_center_between_any_two_sites_preserves_the_state() {
    let n = 6;
    let mut base = mps_after(&crate::circuits::brickwork_circuit(n, 6, 42), 4096);
    base.move_center(n - 1);
    let reference = base.export_statevector().unwrap();
    let reference_norm = base.pauli_expectation(&[]).unwrap().re;

    for from in 0..n {
        for to in 0..n {
            let mut b = base.clone();
            b.move_center(from);
            b.move_center(to);
            b.assert_gauge(to);

            let norm = b.pauli_expectation(&[]).unwrap().re;
            assert!(
                (norm - reference_norm).abs() < 1e-12,
                "norm {norm} after {from} -> {to}, expected {reference_norm}"
            );
            let v = b.export_statevector().unwrap();
            for (i, (r, x)) in reference.iter().zip(&v).enumerate() {
                assert!(
                    (r - x).norm() < 1e-12,
                    "amplitude {i} moved to {x} from {r} after {from} -> {to}"
                );
            }
        }
    }
}

// Drift: each step refactorizes the site it leaves, so the isometry error is
// that factorization's own and must not accumulate over a long walk. The
// fixture reaches bond 16 under a cap of 32, so no step truncates.
#[test]
fn repeated_center_moves_do_not_degrade_the_isometry() {
    let n = 8;
    let mut b = mps_after(&crate::circuits::brickwork_circuit(n, 8, 42), 32);
    b.move_center(n - 1);
    let reference = b.export_statevector().unwrap();
    let bonds = interior_bonds(&b);
    let first = b.gauge_deviation(n - 1);

    let mut worst = first;
    let mut steps = 0usize;
    for _ in 0..30 {
        for target in (0..n).rev() {
            b.move_center(target);
            worst = worst.max(b.gauge_deviation(target));
        }
        for target in 0..n {
            b.move_center(target);
            worst = worst.max(b.gauge_deviation(target));
        }
        steps += 2 * (n - 1);
    }
    assert_eq!(steps, 420);

    assert!(
        worst <= GAUGE_TOLERANCE,
        "gauge deviation reached {worst:.3e} over {steps} steps, from {first:.3e}"
    );
    assert_eq!(
        interior_bonds(&b),
        bonds,
        "a walk that truncates nothing must leave the bond profile alone"
    );
    let v = b.export_statevector().unwrap();
    for (i, (r, x)) in reference.iter().zip(&v).enumerate() {
        assert!(
            (r - x).norm() < 1e-12,
            "amplitude {i} moved to {x} from {r} over {steps} steps"
        );
    }
}
