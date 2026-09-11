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
fn svd_jacobi_keeps_the_singular_vectors_orthonormal_across_six_decades() {
    // `U0 diag(s) V0^H` with `U0` and `V0` orthonormalized by Gram-Schmidt
    // has spectrum `s` to rounding, and the factors must come back orthonormal
    // at every scale, not only on the leading values.
    let n = 8;
    let unitary = |seed: u64| {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut q: Vec<Complex64> = (0..n * n).map(|_| Complex64::new(next(), next())).collect();
        for j in 0..n {
            for _pass in 0..2 {
                for i in 0..j {
                    let dot: Complex64 = (0..n).map(|r| q[i * n + r].conj() * q[j * n + r]).sum();
                    for r in 0..n {
                        let qi = q[i * n + r];
                        q[j * n + r] -= dot * qi;
                    }
                }
            }
            let norm = l2_norm(&q[j * n..(j + 1) * n]);
            for r in 0..n {
                q[j * n + r] /= norm;
            }
        }
        q
    };
    let u0 = unitary(1);
    let v0 = unitary(2);
    let s: Vec<f64> = (0..n).map(|k| 10f64.powf(-6.0 * k as f64 / 7.0)).collect();
    let mut a = vec![ZERO; n * n];
    for c in 0..n {
        for r in 0..n {
            for k in 0..n {
                a[c * n + r] += u0[k * n + r] * s[k] * v0[k * n + c].conj();
            }
        }
    }

    // The construction itself carries rounding of order `eps * s[0] / s[k]`
    // into the tail, so the spectrum is held to an absolute figure.
    let res = svd_jacobi(&a, n, n);
    for (k, (got, want)) in res.s.iter().zip(&s).enumerate() {
        assert!((got - want).abs() < 1e-12, "s[{k}] = {got} expected {want}");
    }
    for j in 0..n {
        for k in 0..n {
            let u: Complex64 = (0..n)
                .map(|r| res.u[j * res.u_rows + r].conj() * res.u[k * res.u_rows + r])
                .sum();
            let v: Complex64 = (0..n)
                .map(|c| res.vt[j * res.vt_cols + c] * res.vt[k * res.vt_cols + c].conj())
                .sum();
            let delta = if j == k { ONE } else { ZERO };
            assert!((u - delta).norm() < 1e-13, "u[{j}].u[{k}] = {u}");
            assert!((v - delta).norm() < 1e-13, "vt[{j}].vt[{k}] = {v}");
        }
    }
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

// A chain of CX gates off one flipped qubit: every gate runs the two-qubit
// path and no cut carries more than one singular value.
fn classical_cascade(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::X, &[0]);
    for q in 0..n - 1 {
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
        b.establish_center(n - 1);
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
    base.establish_center(n - 1);
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
    b.establish_center(n - 1);
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

// The two write conventions differ only in which site keeps diag(S), so they
// leave the same state under a different gauge: the weight site is the center
// the update leaves behind, and the update picks it from the side the center
// arrives on.
#[test]
fn a_two_site_update_weights_the_side_the_center_travels_toward() {
    let n = 6;
    let left_site = 2;
    let gate = Gate::Cx.matrix_4x4();

    let mut base = mps_after(&crate::circuits::brickwork_circuit(n, 6, 42), 4096);
    base.establish_center(left_site);

    let mut weighted_right = base.clone();
    weighted_right
        .apply_adjacent_two_qubit(&gate, left_site, true)
        .unwrap();
    assert_eq!(weighted_right.center, Some(left_site + 1));

    let mut weighted_left = base.clone();
    weighted_left.move_center(left_site + 1);
    weighted_left
        .apply_adjacent_two_qubit(&gate, left_site, true)
        .unwrap();
    assert_eq!(weighted_left.center, Some(left_site));

    // The kernel factorizes with `svd`, whose isometry is looser than the one a
    // center move writes: the U side reads 5.1e-14 on this fixture against
    // 1.6e-15 for the V dagger side, so both are held to a bound the SVD
    // meets rather than to the move's 1e-14.
    for (center, deviation) in [
        (left_site + 1, weighted_right.gauge_deviation(left_site + 1)),
        (left_site, weighted_left.gauge_deviation(left_site)),
    ] {
        assert!(
            deviation < 1e-12,
            "site {center} carries a gauge deviation of {deviation:.3e}"
        );
    }

    assert_ne!(
        weighted_right.sites[left_site].data, weighted_left.sites[left_site].data,
        "both directions wrote the same left site, so the convention did nothing"
    );

    let expected = weighted_right.export_statevector().unwrap();
    let actual = weighted_left.export_statevector().unwrap();
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).norm() < 1e-12,
            "amplitude {i} reads {a} weighting left against {e} weighting right"
        );
    }
}

#[test]
fn thin_qr_drops_a_dependent_column_and_keeps_the_product() {
    // Column 2 is three times column 0, so the factorization has rank 2 and
    // still has to reproduce all three columns.
    let (rows, cols) = (4usize, 3usize);
    let c0 = [1.0, 2.0, -1.0, 0.5];
    let c1 = [0.0, 1.0, 1.0, -2.0];
    let mut a = vec![ZERO; rows * cols];
    for i in 0..rows {
        a[i] = Complex64::new(c0[i], 0.0);
        a[rows + i] = Complex64::new(c1[i], 0.0);
        a[2 * rows + i] = Complex64::new(3.0 * c0[i], 0.0);
    }

    let mut qr = ThinQr::default();
    qr.factorize(&a, rows, cols);
    assert_eq!(qr.rank, 2);
    for i in 0..qr.rank {
        for j in 0..qr.rank {
            let dot: Complex64 = (0..rows)
                .map(|k| qr.q[i * rows + k].conj() * qr.q[j * rows + k])
                .sum();
            let want = if i == j { ONE } else { ZERO };
            assert!((dot - want).norm() < 1e-14, "column {i} against {j}: {dot}");
        }
    }
    for j in 0..cols {
        for k in 0..rows {
            let got: Complex64 = (0..qr.rank)
                .map(|i| qr.q[i * rows + k] * qr.r[i * cols + j])
                .sum();
            assert!(
                (got - a[j * rows + k]).norm() < 1e-13,
                "column {j} row {k}: {got} against {}",
                a[j * rows + k]
            );
        }
    }
}

#[test]
fn thin_qr_of_a_zero_matrix_is_still_an_isometry() {
    let (rows, cols) = (4usize, 2usize);
    let mut qr = ThinQr::default();

    // On buffers a dense factorization has already filled, since that is how
    // the walk reaches this: the kept column reads a norm of 1.4 if the zero
    // case takes what it finds there.
    let dense: Vec<Complex64> = (0..rows * cols)
        .map(|i| Complex64::new(i as f64 + 1.0, 0.5))
        .collect();
    qr.factorize(&dense, rows, cols);
    assert_eq!(qr.rank, 2);

    qr.factorize(&vec![ZERO; rows * cols], rows, cols);
    assert_eq!(qr.rank, 1);
    assert!((l2_norm(&qr.q[..rows]) - 1.0).abs() < 1e-15);
    assert!(qr.r[..cols].iter().all(|x| x.norm() == 0.0));
}

// Squared 2-norm distance between two chains over one site layout, taken on
// the stored tensors: a truncated chain is not normalized and every read
// rescales, which would hide the very weight under test.
fn squared_distance(a: &MpsBackend, b: &MpsBackend) -> f64 {
    a.pauli_expectation(&[]).unwrap().re - 2.0 * a.inner_product(b).unwrap().re
        + b.pauli_expectation(&[]).unwrap().re
}

// Apply `gate` to `base` twice, once with room for every singular value and
// once under `cap`, and return what the capped run booked against the distance
// it moved the state.
//
// The booked figure is the fraction of the cut's weight that went, so on a
// chain whose norm a projection or a Kraus branch has already taken below one
// it is that fraction of the norm that the distance can be compared against.
fn one_cut(base: &MpsBackend, cap: usize, gate: impl Fn(&mut MpsBackend)) -> (f64, f64) {
    let mut kept = base.clone();
    kept.max_bond_dim = usize::MAX;
    kept.svd_epsilon = 0.0;
    kept.reset_truncation_tracking();
    gate(&mut kept);
    assert!(
        kept.truncation_discarded() < 1e-30,
        "the reference run lost {:.3e}",
        kept.truncation_discarded()
    );

    let mut cut = base.clone();
    cut.max_bond_dim = cap;
    cut.reset_truncation_tracking();
    gate(&mut cut);

    let norm = base.pauli_expectation(&[]).unwrap().re;
    (
        cut.truncation_discarded() * norm,
        squared_distance(&kept, &cut),
    )
}

fn cx_at(left_site: usize) -> impl Fn(&mut MpsBackend) {
    move |b: &mut MpsBackend| {
        b.apply_adjacent_two_qubit(&Gate::Cx.matrix_4x4(), left_site, true)
            .unwrap();
    }
}

// The property the center exists for: against an orthonormal environment the
// weight a cut drops is the squared 2-norm distance it moves the state, so the
// number the cut books is the error it made rather than a bound on it. Held per
// cut, since the strict bound over a sequence is the square of the summed
// square roots and a whole-circuit version would test that looser claim
// instead. Without the center, cap 3 here books 0.28 against a realized 0.13.
#[test]
fn a_capped_cut_books_the_error_it_makes() {
    let base = mps_after(&crate::circuits::brickwork_circuit(8, 6, 42), 4096);
    for cap in [2usize, 3, 5] {
        let (booked, realized) = one_cut(&base, cap, cx_at(3));
        assert!(booked > 1e-3, "cap {cap} truncated nothing to check");
        assert!(
            (realized - booked).abs() < 1e-14,
            "cap {cap} booked {booked:.15e} and moved the state {realized:.15e}"
        );
    }
}

// The gate predicate is the rank of the cut against the cap: a pair yields
// `2 * bl.min(br)` singular values, so a chain whose bonds keep that under the
// cap can lose nothing and takes the path it took before, untouched.
#[test]
fn a_chain_under_the_cap_stays_off_the_walk() {
    for (label, circuit, cap, peak) in [
        ("cascade_6", classical_cascade(6), 32, 1),
        ("bell_pairs_6", bell_pairs(6), 32, 2),
        (
            "brickwork_10_d4",
            crate::circuits::brickwork_circuit(10, 4, 42),
            256,
            4,
        ),
    ] {
        let b = mps_after(&circuit, cap);
        assert_eq!(b.current_max_bond_dim(), peak, "{label} bond peak");
        assert_eq!(
            b.center, None,
            "{label} recorded a center under a cap of {cap}"
        );
        assert_eq!(
            b.center_steps, 0,
            "{label} walked the chain under a cap of {cap}"
        );
    }

    // The boundary, so the assertion above is a predicate and not an accident:
    // a rank-4 cut clears a cap of 4 and does not clear a cap of 3.
    let circuit = crate::circuits::brickwork_circuit(10, 2, 42);
    assert_eq!(mps_after(&circuit, 4).current_max_bond_dim(), 2);
    assert_eq!(mps_after(&circuit, 4).center, None);
    assert!(mps_after(&circuit, 3).center.is_some());
}

// A center that claims more than the chain has is worse than none: a move
// repairs the span it walks and leaves the rest wrong. Check every step of a
// run rather than the end of one.
#[test]
fn the_gauge_holds_through_a_circuit_that_drives_the_policy() {
    let n = 8;
    let circuit = crate::circuits::brickwork_circuit(n, 10, 42);
    let mut b = MpsBackend::new(42, 8);
    b.init(n, 0).unwrap();

    let mut checked = 0usize;
    for instruction in &circuit.instructions {
        b.apply(instruction).unwrap();
        if let Some(center) = b.center {
            // The kernel factorizes with `svd`, whose isometry is looser than
            // the walk's QR, so this is the bound the SVD meets rather than the
            // 1e-14 a move is held to.
            let worst = b.gauge_deviation(center);
            assert!(
                worst < 1e-12,
                "gauge deviation {worst:.3e} about center {center}"
            );
            checked += 1;
        }
    }

    assert!(
        checked > 50,
        "the policy engaged for {checked} of {} instructions",
        circuit.instructions.len()
    );
    assert!(b.truncation_discarded() > 0.0, "the cap never bit");
}

fn chain_with_center(center: usize) -> MpsBackend {
    let mut b = MpsBackend::new(42, 4096);
    b.init(6, 1).unwrap();
    b.apply_instructions(&crate::circuits::brickwork_circuit(6, 6, 42).instructions)
        .unwrap();
    b.establish_center(center);
    b
}

// A write that is not an isometry breaks the gauge on a site the record claims
// one for, and a later move would repair the span it walks and leave that site
// wrong. Drop the record instead, so the next cut rebuilds.
#[test]
fn a_non_unitary_write_off_the_center_drops_it() {
    let center = 2;
    let damping = [[ONE, ZERO], [ZERO, Complex64::new(0.5, 0.0)]];

    let mut kraus = chain_with_center(center);
    kraus.apply_1q_matrix(center + 1, &damping).unwrap();
    assert_eq!(kraus.center, None, "a Kraus branch kept the record");

    let mut measured = chain_with_center(center);
    measured
        .apply(&Instruction::Measure {
            qubit: center + 1,
            classical_bit: 0,
        })
        .unwrap();
    assert_eq!(measured.center, None, "a measurement kept the record");

    let mut reset = chain_with_center(center);
    reset.reset(center + 1).unwrap();
    assert_eq!(reset.center, None, "a reset kept the record");

    // What dropping it buys: the cut after a Kraus branch still books the error
    // it makes, because it rebuilds rather than trusting a stale record.
    let (booked, realized) = one_cut(&kraus, 3, cx_at(3));
    assert!(booked > 1e-3, "the cut truncated nothing to check");
    assert!(
        (realized - booked).abs() < 1e-14,
        "after a Kraus branch the cut booked {booked:.15e} and moved the state {realized:.15e}"
    );
}

// The center site is under no isometry claim, so a projection there leaves
// every other site exactly as canonical as it was.
#[test]
fn a_projection_on_the_center_keeps_it() {
    let center = 2;
    let mut b = chain_with_center(center);
    b.apply(&Instruction::Measure {
        qubit: center,
        classical_bit: 0,
    })
    .unwrap();

    assert_eq!(b.center, Some(center));
    b.assert_gauge(center);
}

// End to end: a chain of Schmidt rank 2 under a cap of 3 drives the policy on
// every pair away from the ends, while the state itself never loses a singular
// value, which leaves the comparison exact rather than tolerant of truncation.
#[test]
fn a_policy_driven_circuit_matches_the_statevector() {
    let n = 6;
    let mut circuit = Circuit::new(n, 0);
    circuit.add_gate(Gate::Ry(0.7), &[0]);
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Rz(0.3 + q as f64), &[q]);
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }

    let b = mps_after(&circuit, 3);
    assert!(b.center.is_some(), "the fixture never drove the policy");
    assert_eq!(b.current_max_bond_dim(), 2);
    // The walk is exact and books nothing, which the CAMPS T-gate path reads
    // as a hard error when it is not so.
    assert!(b.center_steps > 0, "the walk never moved the center");
    assert_eq!(b.truncation_discarded(), 0.0);

    let mut sv = crate::backend::statevector::StatevectorBackend::new(42);
    sv.init(n, 0).unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();

    let expected = sv.export_statevector().unwrap();
    let actual = b.export_statevector().unwrap();
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).norm() < 1e-15,
            "amplitude {i} reads {a} against {e}"
        );
    }
}

// The parking rule is what makes the policy affordable: the update leaves the
// center on the far side of the pair in the direction of travel, so a run of
// adjacent gates carries it along without a factorization of its own.
#[test]
fn a_run_of_adjacent_gates_carries_the_center_along() {
    let n = 10;
    let mut b = MpsBackend::new(42, 4);
    b.init(n, 0).unwrap();
    b.apply_instructions(&crate::circuits::brickwork_circuit(n, 6, 42).instructions)
        .unwrap();
    assert!(b.center.is_some(), "the fixture never drove the policy");

    // Rightward: the run pays the reach to the first pair and one step to turn
    // the center around, after which each pair already has it on its left.
    let mark = b.center_steps;
    for q in 0..n - 1 {
        b.dispatch_gate(&Gate::Cx, &[q, q + 1]).unwrap();
    }
    assert_eq!(
        b.center_steps - mark,
        n - 2,
        "rightward run of {} gates",
        n - 1
    );

    // Leftward: the pair the center already sits on takes the weight on its
    // left instead, which turns the run around for nothing.
    let mark = b.center_steps;
    for q in (0..n - 1).rev() {
        b.dispatch_gate(&Gate::Cx, &[q, q + 1]).unwrap();
    }
    assert_eq!(b.center_steps - mark, 0, "leftward run of {} gates", n - 1);

    // A routed hop is monotone as well: the swaps march the pair together from
    // the far end inward, so the center pays the jump to the first swap and
    // nothing after it.
    assert_eq!(b.center, Some(0));
    let mark = b.center_steps;
    b.dispatch_gate(&Gate::Cx, &[1, 7]).unwrap();
    assert_eq!(b.center_steps - mark, 7, "hop from site 1 to site 7");
}

// A block gate decomposes left to right and leaves the weight on the last site
// of the block, so the center ends there and the walk owes only the distance
// into the block.
#[test]
fn a_block_gate_leaves_the_center_at_the_block_end() {
    use crate::gates::McuData;

    let n = 10;
    let mut b = MpsBackend::new(42, 4);
    b.init(n, 0).unwrap();
    b.apply_instructions(&crate::circuits::brickwork_circuit(n, 6, 42).instructions)
        .unwrap();
    // From the left of the block, so the walk into it and the re-point after
    // it land on different sites: a center left at the near edge reads a gauge
    // deviation of 9.6e-1 there.
    b.move_center(1);

    let mark = b.center_steps;
    let mat = [[ZERO, ONE], [ONE, ZERO]];
    b.dispatch_gate(
        &Gate::Mcu(Box::new(McuData {
            num_controls: 2,
            mat,
        })),
        &[3, 4, 5],
    )
    .unwrap();

    let center = b.center.expect("the block gate dropped the center");
    assert_eq!(center, 5);
    assert_eq!(b.center_steps - mark, 2, "the walk stops at the near edge");
    let worst = b.gauge_deviation(center);
    assert!(
        worst < 1e-13,
        "gauge deviation {worst:.3e} after a block gate"
    );
}

// Bubble routing makes its own kernel calls rather than going through the
// two-qubit entry point, so the policy has to reach it there.
#[test]
fn bubble_routing_keeps_the_center() {
    use crate::gates::BatchPhaseData;

    let n = 8;
    let mut b = MpsBackend::new(42, 4);
    b.init(n, 0).unwrap();
    b.apply_instructions(&crate::circuits::brickwork_circuit(n, 6, 42).instructions)
        .unwrap();
    assert!(b.center.is_some(), "the fixture never drove the policy");

    b.dispatch_gate(
        &Gate::BatchPhase(Box::new(BatchPhaseData {
            phases: smallvec::smallvec![
                (1, Complex64::from_polar(1.0, 0.5)),
                (5, Complex64::from_polar(1.0, 1.2)),
                (6, Complex64::from_polar(1.0, 2.1)),
            ],
        })),
        &[3],
    )
    .unwrap();

    let center = b.center.expect("bubble routing dropped the center");
    let worst = b.gauge_deviation(center);
    assert!(
        worst < 1e-13,
        "gauge deviation {worst:.3e} about center {center}"
    );
}

fn mcu_at(control_pair: [usize; 3]) -> impl Fn(&mut MpsBackend) {
    use crate::gates::McuData;

    let gate = Gate::Mcu(Box::new(McuData {
        num_controls: 2,
        mat: [[ZERO, ONE], [ONE, ZERO]],
    }));
    move |b: &mut MpsBackend| {
        b.dispatch_gate(&gate, &control_pair).unwrap();
    }
}

// The gauge is not bookkeeping. A cut against a non-orthogonal environment
// keeps a different subspace, so it lands further from the state the uncapped
// run holds: 2.0x, 1.5x and 4.8x further on the three caps below, and 1.8x on
// the block gate. A center claimed but not held is the only way left to reach
// such a cut, which is why the invalidation rules exist.
#[test]
fn a_centered_cut_lands_closer_than_one_in_the_wrong_gauge() {
    let base = mps_after(&crate::circuits::brickwork_circuit(8, 6, 42), 4096);
    let mut stale = base.clone();
    stale.center = Some(3);

    for cap in [2usize, 3, 5] {
        let (_, centered) = one_cut(&base, cap, cx_at(3));
        let (_, ungauged) = one_cut(&stale, cap, cx_at(3));
        assert!(
            centered < 0.9 * ungauged,
            "cap {cap} moved the state {centered:.6e} centered against {ungauged:.6e} ungauged"
        );
    }

    let (_, centered) = one_cut(&base, 3, mcu_at([3, 4, 5]));
    let (_, ungauged) = one_cut(&stale, 3, mcu_at([3, 4, 5]));
    assert!(
        centered < 0.9 * ungauged,
        "the block gate moved the state {centered:.6e} centered against {ungauged:.6e} ungauged"
    );
}

// A raised threshold cuts real weight on a chain whose bonds never approach
// the cap, so the cap alone does not decide whether the gauge matters. At 0.2
// this cut books 1.700606e-2 and moves the state by the same, where an
// ungauged one books 6.900e-3 against a realized 9.587e-3, understating by
// 28% the error it made.
#[test]
fn a_raised_epsilon_gauges_a_chain_under_the_cap() {
    let mut base = mps_after(&crate::circuits::brickwork_circuit(8, 6, 42), 4096);
    base.set_svd_epsilon(0.2);

    let (booked, realized) = one_cut(&base, 4096, cx_at(3));
    assert!(booked > 1e-3, "the raised threshold cut nothing to check");
    assert!(
        (realized - booked).abs() < 1e-14,
        "booked {booked:.15e} against a realized {realized:.15e}"
    );

    // The construction default sheds at most rank * epsilon^2, which is under
    // rounding, so it leaves the chain on the unchanged path.
    let default = mps_after(&crate::circuits::brickwork_circuit(8, 6, 42), 4096);
    assert_eq!(default.center, None);
    assert_eq!(default.center_steps, 0);
}

// The block decomposition truncates at every one of its cuts, so it needs a
// trigger of its own rather than only inheriting a center the two-qubit path
// left behind.
#[test]
fn a_block_gate_establishes_a_center_of_its_own() {
    let base = mps_after(&crate::circuits::brickwork_circuit(8, 6, 42), 4096);
    assert_eq!(base.center, None);

    let (booked, realized) = one_cut(&base, 3, mcu_at([3, 4, 5]));

    let mut cut = base.clone();
    cut.max_bond_dim = 3;
    mcu_at([3, 4, 5])(&mut cut);
    assert_eq!(cut.center, Some(5));
    assert!(cut.center_steps > 0, "the block gate never walked");

    // Three sites decompose in two cuts and the total sums both, so it answers
    // for the accumulation rather than for one cut: 1.514759e-1 booked against
    // a realized 1.457433e-1, over rather than under.
    assert!(
        booked >= realized && booked < 1.1 * realized,
        "the block booked {booked:.6e} against a realized {realized:.6e}"
    );
}

// End to end, which is what a caller sees: the error the whole run carries
// against the untruncated chain, and how close the reported total lands to it.
// The ungauged path read an infidelity of 3.566e-2 here against a booked
// 5.463e-2, so the state is 258 times further out than this one and the figure
// describing it misses by 53%.
#[test]
fn a_capped_run_lands_where_it_says_it_does() {
    let circuit = crate::circuits::brickwork_circuit(14, 24, 0xDEAD_BEEF);
    let n = circuit.num_qubits;

    let mut exact = MpsBackend::new(42, 1 << 20);
    exact.init(n, 0).unwrap();
    exact.apply_instructions(&circuit.instructions).unwrap();
    assert!(
        exact.truncation_discarded() < 1e-20,
        "the reference truncated"
    );

    let mut capped = MpsBackend::new(42, 64);
    capped.init(n, 0).unwrap();
    capped.apply_instructions(&circuit.instructions).unwrap();

    let overlap = exact.inner_product(&capped).unwrap().norm_sqr();
    let infidelity = 1.0
        - overlap
            / (exact.pauli_expectation(&[]).unwrap().re
                * capped.pauli_expectation(&[]).unwrap().re);
    assert!(
        infidelity < 1e-3,
        "the capped run sits {infidelity:.6e} from the untruncated one"
    );

    // The total sums one figure per cut, so it answers for the accumulation
    // and is not owed exactness here, only the right size.
    let booked = capped.truncation_discarded();
    assert!(
        (infidelity - booked).abs() < 0.1 * booked,
        "booked {booked:.6e} against a realized {infidelity:.6e}"
    );
}

// The middle bond binds where the pair bonds do not: a chain carrying a bond
// wider than the rank feeding it cannot lose weight the wider bond suggests.
#[test]
fn the_middle_bond_caps_the_rank_a_two_site_cut_carries() {
    assert_eq!(cut_rank(20, 2, 20), 8);
    assert_eq!(cut_rank(20, 64, 20), 40);
    assert_eq!(cut_rank(1, 64, 64), 2);
    assert_eq!(cut_rank(64, 16, 8), 16);
}

// Eight sites at a cap nothing reaches, so the walk is on every pair once a
// center is recorded and no cut discards more than rounding, which leaves
// two gate orders agreeing to rounding rather than to the truncation error.
fn chain_with_center_at_the_right_end() -> MpsBackend {
    let n = 8;
    let mut b = MpsBackend::new(42, 4096);
    b.init(n, 1).unwrap();
    b.apply_instructions(&crate::circuits::brickwork_circuit(n, 6, 42).instructions)
        .unwrap();
    b.establish_center(n - 1);
    assert!(b.truncation_discarded() < 1e-30);
    b
}

// Brick layers with fixed angles: rotations on every qubit, then an
// entangling gate on each pair of the parity the layer index sets, written
// left to right or, on `snaked` layers, right to left.
fn brick_layers(
    n: usize,
    depth: usize,
    entangler: impl Fn(usize) -> Gate,
    snaked: impl Fn(usize) -> bool,
) -> Circuit {
    let mut c = Circuit::new(n, 1);
    for layer in 0..depth {
        for q in 0..n {
            let angle = 0.1 + 0.37 * (layer * n + q) as f64;
            c.add_gate(Gate::Ry(angle), &[q]);
            c.add_gate(Gate::Rz(angle * 0.5), &[q]);
        }
        let mut pairs: Vec<usize> = (layer % 2..n - 1).step_by(2).collect();
        if snaked(layer) {
            pairs.reverse();
        }
        for q in pairs {
            c.add_gate(entangler(q), &[q, q + 1]);
        }
    }
    c
}

fn applied_one_at_a_time(mut b: MpsBackend, circuit: &Circuit) -> MpsBackend {
    for instruction in &circuit.instructions {
        b.apply(instruction).unwrap();
    }
    b
}

fn applied_as_a_batch(mut b: MpsBackend, circuit: &Circuit) -> MpsBackend {
    b.apply_instructions(&circuit.instructions).unwrap();
    b
}

fn assert_chains_identical(a: &MpsBackend, b: &MpsBackend, label: &str) {
    assert_eq!(a.center, b.center, "{label}: center");
    assert_eq!(a.center_steps, b.center_steps, "{label}: center steps");
    for (site, (x, y)) in a.sites.iter().zip(&b.sites).enumerate() {
        assert_eq!(
            (x.bond_left, x.bond_right),
            (y.bond_left, y.bond_right),
            "{label}: site {site} shape"
        );
        assert!(x.data == y.data, "{label}: site {site} data differs");
    }
}

// The snake: with the center at the right end, the even layers here start
// from their last pair and the odd ones from their first, so each layer costs
// its interior steps and nothing to reach it. The batch must produce exactly
// the run that the snaked circuit produces gate by gate, and the state the
// written order produces up to rounding.
#[test]
fn a_brick_layer_enters_from_the_end_the_center_is_at() {
    let warm = chain_with_center_at_the_right_end();
    let n = warm.num_qubits;
    let written = brick_layers(n, 4, |_| Gate::Cz, |_| false);
    let snaked = brick_layers(n, 4, |_| Gate::Cz, |layer| layer % 2 == 0);

    let reordered = applied_as_a_batch(warm.clone(), &written);
    let reference = applied_one_at_a_time(warm.clone(), &snaked);
    let plain = applied_one_at_a_time(warm.clone(), &written);

    // Four layers of three interior steps, plus one on the third: it starts
    // with the center on the left site of its last pair, so the update parks
    // the weight on the right and the walk steps back across it. Written
    // order pays the width of the chain back to the first pair on every
    // layer.
    assert_eq!(reference.center_steps - warm.center_steps, 13);
    assert_eq!(plain.center_steps - warm.center_steps, 35);
    assert_chains_identical(&reordered, &reference, "batch against snaked");

    let expected = plain.export_statevector().unwrap();
    let actual = reordered.export_statevector().unwrap();
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).norm() < 1e-13,
            "amplitude {i} reads {a} against {e}"
        );
    }
}

// Anything that is not such a gate ends a run where it stands, so a divider
// inside a layer that the walk would otherwise enter from the far end leaves
// the layer applied as written on both sides of it.
#[test]
fn a_run_does_not_cross_a_barrier() {
    use crate::circuit::{ClassicalCondition, SmallVec};

    let warm = chain_with_center_at_the_right_end();
    let n = warm.num_qubits;
    let layer = |divider: Option<Instruction>| {
        let mut c = Circuit::new(n, 1);
        c.add_gate(Gate::Cz, &[0, 1]);
        if let Some(divider) = divider {
            c.instructions.push(divider);
        }
        for q in (2..n - 1).step_by(2) {
            c.add_gate(Gate::Cz, &[q, q + 1]);
        }
        c
    };

    let whole = layer(None);
    assert_eq!(
        applied_as_a_batch(warm.clone(), &whole).center_steps - warm.center_steps,
        3,
        "the undivided layer is the fixture the reorder fires on"
    );
    assert_eq!(
        applied_one_at_a_time(warm.clone(), &whole).center_steps - warm.center_steps,
        10
    );

    let dividers: Vec<(&str, Instruction)> = vec![
        (
            "barrier",
            Instruction::Barrier {
                qubits: SmallVec::from_slice(&[0, 1]),
            },
        ),
        (
            "measure",
            Instruction::Measure {
                qubit: 0,
                classical_bit: 0,
            },
        ),
        ("reset", Instruction::Reset { qubit: 0 }),
        (
            "conditional",
            Instruction::Conditional {
                condition: ClassicalCondition::BitIsOne(0),
                gate: Gate::X,
                targets: SmallVec::from_slice(&[0]),
            },
        ),
        (
            "region",
            crate::circuit::guarded(
                ClassicalCondition::BitIsOne(0),
                vec![
                    Instruction::Gate {
                        gate: Gate::X,
                        targets: SmallVec::from_slice(&[0]),
                    },
                    Instruction::Reset { qubit: 0 },
                ],
            )
            .unwrap(),
        ),
        (
            "rotation",
            Instruction::Gate {
                gate: Gate::Ry(0.3),
                targets: SmallVec::from_slice(&[n - 1]),
            },
        ),
        (
            "routed pair",
            Instruction::Gate {
                gate: Gate::Cz,
                targets: SmallVec::from_slice(&[0, n - 1]),
            },
        ),
    ];
    for (label, divider) in dividers {
        let divided = layer(Some(divider));
        let batch = applied_as_a_batch(warm.clone(), &divided);
        let one_at_a_time = applied_one_at_a_time(warm.clone(), &divided);
        assert_chains_identical(&batch, &one_at_a_time, label);
        assert!(
            batch.center_steps - warm.center_steps >= 6,
            "{label}: the run crossed the divider"
        );
    }
}

// A sequence that is not a run of disjoint adjacent pairs with increasing
// left sites is applied as written: overlapping pairs, pairs written right to
// left, and pairs that need routing.
#[test]
fn a_sequence_that_is_not_a_brick_layer_is_applied_as_written() {
    let warm = chain_with_center_at_the_right_end();
    let n = warm.num_qubits;

    let mut ladder = Circuit::new(n, 1);
    for q in 0..n - 1 {
        ladder.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let mut leftward = Circuit::new(n, 1);
    for q in (0..n - 1).rev().step_by(2) {
        leftward.add_gate(Gate::Cz, &[q, q + 1]);
    }
    let mut hop_first = Circuit::new(n, 1);
    hop_first.add_gate(Gate::Cz, &[0, 5]);
    for q in (2..n - 1).step_by(2) {
        hop_first.add_gate(Gate::Cz, &[q, q + 1]);
    }
    let matched = crate::circuits::matched_brickwork_circuit(n, 4, 42);

    for (label, circuit) in [
        ("ladder", &ladder),
        ("leftward", &leftward),
        ("hop first", &hop_first),
        ("matched", &matched),
    ] {
        let batch = applied_as_a_batch(warm.clone(), circuit);
        let one_at_a_time = applied_one_at_a_time(warm.clone(), circuit);
        assert_chains_identical(&batch, &one_at_a_time, label);
    }
}

// The fused forms carry the same layers as lists inside one gate, so they
// take the same walk and land on the same bits as the written gates.
#[test]
fn fused_pair_lists_take_the_same_walk() {
    use crate::circuit::SmallVec;
    use crate::gates::{BatchRzzData, Multi2qData};

    let warm = chain_with_center_at_the_right_end();
    let n = warm.num_qubits;
    let angle = |q: usize| 0.2 + 0.11 * q as f64;
    let written = brick_layers(n, 4, |q| Gate::Rzz(angle(q)), |_| false);

    let mut multi = Circuit::new(n, 1);
    let mut batched = Circuit::new(n, 1);
    let mut gates = Vec::new();
    let mut edges = Vec::new();
    let mut qubits: SmallVec<[usize; 4]> = SmallVec::new();
    let flush = |gates: &mut Vec<_>, edges: &mut Vec<_>, qubits: &mut SmallVec<[usize; 4]>| {
        let mut out = Vec::new();
        if !gates.is_empty() {
            let data = Multi2qData {
                gates: std::mem::take(gates),
            };
            out.push(Instruction::Gate {
                gate: Gate::Multi2q(Box::new(data)),
                targets: qubits.clone(),
            });
            let data = BatchRzzData {
                edges: std::mem::take(edges),
            };
            out.push(Instruction::Gate {
                gate: Gate::BatchRzz(Box::new(data)),
                targets: std::mem::take(qubits),
            });
        }
        out
    };
    for instruction in &written.instructions {
        match instruction {
            Instruction::Gate {
                gate: Gate::Rzz(theta),
                targets,
            } => {
                gates.push((targets[0], targets[1], Gate::Rzz(*theta).matrix_4x4()));
                edges.push((targets[0], targets[1], *theta));
                qubits.extend_from_slice(targets);
            }
            other => {
                if let [m, b] = flush(&mut gates, &mut edges, &mut qubits).as_slice() {
                    multi.instructions.push(m.clone());
                    batched.instructions.push(b.clone());
                }
                multi.instructions.push(other.clone());
                batched.instructions.push(other.clone());
            }
        }
    }
    if let [m, b] = flush(&mut gates, &mut edges, &mut qubits).as_slice() {
        multi.instructions.push(m.clone());
        batched.instructions.push(b.clone());
    }

    let reordered = applied_as_a_batch(warm.clone(), &written);
    assert_eq!(reordered.center_steps - warm.center_steps, 13);
    let multi = applied_as_a_batch(warm.clone(), &multi);
    assert_chains_identical(&multi, &reordered, "multi2q");
    let batched = applied_as_a_batch(warm.clone(), &batched);
    assert_chains_identical(&batched, &reordered, "batch rzz");
}

// An overlapping pair list inside a fused payload is applied as written as
// well: the scan inside the arm stops at the second pair, so the list lands
// on the same bits as the gates applied one at a time.
#[test]
fn an_overlapping_pair_list_in_a_fused_payload_is_applied_as_written() {
    use crate::circuit::SmallVec;
    use crate::gates::{BatchRzzData, Multi2qData};

    let warm = chain_with_center_at_the_right_end();
    let n = warm.num_qubits;
    let qubits: SmallVec<[usize; 4]> = (0..n).collect();

    let mut ladder = Circuit::new(n, 1);
    let mut gates = Vec::new();
    for q in 0..n - 1 {
        ladder.add_gate(Gate::Cx, &[q, q + 1]);
        gates.push((q, q + 1, Gate::Cx.matrix_4x4()));
    }
    let mut multi = warm.clone();
    multi
        .apply(&Instruction::Gate {
            gate: Gate::Multi2q(Box::new(Multi2qData { gates })),
            targets: qubits.clone(),
        })
        .unwrap();
    assert_chains_identical(
        &multi,
        &applied_one_at_a_time(warm.clone(), &ladder),
        "multi2q",
    );

    let mut rzz_ladder = Circuit::new(n, 1);
    let mut edges = Vec::new();
    for q in 0..n - 1 {
        let theta = 0.3 + 0.2 * q as f64;
        rzz_ladder.add_gate(Gate::Rzz(theta), &[q, q + 1]);
        edges.push((q, q + 1, theta));
    }
    let mut batched = warm.clone();
    batched
        .apply(&Instruction::Gate {
            gate: Gate::BatchRzz(Box::new(BatchRzzData { edges })),
            targets: qubits,
        })
        .unwrap();
    assert_chains_identical(
        &batched,
        &applied_one_at_a_time(warm.clone(), &rzz_ladder),
        "batch rzz",
    );
}

// The snake changes which cuts truncate against which environment, so the
// realized error of a capped run must land where the written order lands it,
// not merely where the booked total says.
#[test]
fn the_snake_loses_no_more_than_the_written_order() {
    let circuit = crate::circuits::brickwork_circuit(18, 24, 0xDEAD_BEEF);
    let n = circuit.num_qubits;

    let mut exact = MpsBackend::new(42, 1 << 20);
    exact.init(n, 0).unwrap();
    exact.apply_instructions(&circuit.instructions).unwrap();
    assert!(
        exact.truncation_discarded() < 1e-20,
        "the reference truncated"
    );
    let exact_norm = exact.pauli_expectation(&[]).unwrap().re;

    let infidelity = |capped: &MpsBackend| {
        let overlap = exact.inner_product(capped).unwrap().norm_sqr();
        1.0 - overlap / (exact_norm * capped.pauli_expectation(&[]).unwrap().re)
    };

    let mut snake = MpsBackend::new(42, 64);
    snake.init(n, 0).unwrap();
    snake.apply_instructions(&circuit.instructions).unwrap();
    let mut written = MpsBackend::new(42, 64);
    written.init(n, 0).unwrap();
    for instruction in &circuit.instructions {
        written.apply(instruction).unwrap();
    }
    assert!(
        snake.center_steps < written.center_steps,
        "the snake never fired"
    );

    let snake_error = infidelity(&snake);
    let written_error = infidelity(&written);
    assert!(
        snake_error <= 1.05 * written_error,
        "the snake sits {snake_error:.6e} from the reference against {written_error:.6e}"
    );
}

// Fusion hands a brick layer over in pieces, a `Multi2q` with the leftover
// pairs as `Fused2q` gates or a second list, so a run has to be counted in
// gate entries across the instructions rather than in instructions.
#[test]
fn a_brick_layer_split_across_fused_instructions_still_enters_from_the_near_end() {
    use crate::circuit::SmallVec;
    use crate::gates::Multi2qData;

    let warm = chain_with_center_at_the_right_end();
    let n = warm.num_qubits;
    let written = brick_layers(n, 4, |_| Gate::Cz, |_| false);

    let mut split = Circuit::new(n, 1);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut layer = 0;
    let list = |pairs: &[(usize, usize)]| Instruction::Gate {
        gate: Gate::Multi2q(Box::new(Multi2qData {
            gates: pairs
                .iter()
                .map(|&(q0, q1)| (q0, q1, Gate::Cz.matrix_4x4()))
                .collect(),
        })),
        targets: pairs.iter().flat_map(|&(q0, q1)| [q0, q1]).collect(),
    };
    let fused = |(q0, q1): (usize, usize)| Instruction::Gate {
        gate: Gate::Fused2q(Box::new(Gate::Cz.matrix_4x4())),
        targets: SmallVec::from_slice(&[q0, q1]),
    };
    let flush = |pairs: &mut Vec<(usize, usize)>, layer: &mut usize, out: &mut Circuit| {
        if pairs.is_empty() {
            return;
        }
        let m = pairs.len();
        if layer.is_multiple_of(2) {
            out.instructions.push(list(&pairs[..m - 2]));
            out.instructions.push(fused(pairs[m - 2]));
            out.instructions.push(fused(pairs[m - 1]));
        } else {
            out.instructions.push(list(&pairs[..m - 1]));
            out.instructions.push(list(&pairs[m - 1..]));
        }
        pairs.clear();
        *layer += 1;
    };
    for instruction in &written.instructions {
        match instruction {
            Instruction::Gate {
                gate: Gate::Cz,
                targets,
            } => pairs.push((targets[0], targets[1])),
            other => {
                flush(&mut pairs, &mut layer, &mut split);
                split.instructions.push(other.clone());
            }
        }
    }
    flush(&mut pairs, &mut layer, &mut split);
    assert_eq!(layer, 4);

    let reordered = applied_as_a_batch(warm.clone(), &written);
    assert_eq!(reordered.center_steps - warm.center_steps, 13);
    let from_pieces = applied_as_a_batch(warm.clone(), &split);
    assert_chains_identical(&from_pieces, &reordered, "split layers");
}
