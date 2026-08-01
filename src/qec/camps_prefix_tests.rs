use super::*;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::{Instruction, SmallVec};
use num_complex::Complex64;

fn eval_z(
    prefix: &SignedCliffordPrefix,
    mps: &crate::backend::mps::MpsBackend,
    qubits: &[usize],
) -> f64 {
    let terms: Vec<_> = qubits
        .iter()
        .copied()
        .map(crate::sim::unified_pauli::PauliTerm::z)
        .collect();
    evaluate_pauli_observable_camps(prefix, mps, &terms).unwrap()
}

fn pauli_action_on_basis(p: &SignedPauli, i: usize, n: usize) -> (Complex64, usize) {
    let mut j = i;
    let mut phase4: u32 = u32::from(p.phase4);
    for q in 0..n {
        let b = (i >> q) & 1;
        match p.pauli_at(q) {
            PauliKind::I => {}
            PauliKind::X => {
                j ^= 1 << q;
            }
            PauliKind::Y => {
                j ^= 1 << q;
                phase4 = phase4.wrapping_add(if b == 0 { 1 } else { 3 });
            }
            PauliKind::Z => {
                if b == 1 {
                    phase4 = phase4.wrapping_add(2);
                }
            }
        }
    }
    let phase = match phase4 & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    (phase, j)
}

fn pauli_string_expectation(state: &[Complex64], n: usize, p: &SignedPauli) -> Complex64 {
    let dim = 1usize << n;
    assert_eq!(state.len(), dim);
    let mut acc = Complex64::new(0.0, 0.0);
    for i in 0..dim {
        let (phase, j) = pauli_action_on_basis(p, i, n);
        acc += state[j].conj() * (phase * state[i]);
    }
    acc
}

fn run_gates(n: usize, gates: &[(Gate, Vec<usize>)]) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(0);
    backend.init(n, 0).unwrap();
    let instrs: Vec<Instruction> = gates
        .iter()
        .map(|(g, t)| Instruction::Gate {
            gate: g.clone(),
            targets: SmallVec::from_slice(t),
        })
        .collect();
    backend.apply_instructions(&instrs).unwrap();
    backend.export_statevector().unwrap()
}

fn randomizing_prefix(n: usize, seed: u64) -> Vec<(Gate, Vec<usize>)> {
    use rand::RngExt;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut g = vec![(Gate::H, vec![0])];
    for q in 0..n {
        g.push((Gate::Ry(rng.random_range(0.3..2.7)), vec![q]));
        g.push((Gate::Rz(rng.random_range(0.3..2.7)), vec![q]));
        if q + 1 < n {
            g.push((Gate::Cx, vec![q, q + 1]));
        }
        g.push((Gate::Rx(rng.random_range(0.3..2.7)), vec![q]));
    }
    g
}

fn assert_consistent(n: usize, gates: &[(Gate, Vec<usize>)], seed: u64) {
    let pre_gates = randomizing_prefix(n, seed);
    let pre = run_gates(n, &pre_gates);
    let mut combined = pre_gates;
    combined.extend_from_slice(gates);
    let post = run_gates(n, &combined);

    let mut prefix = SignedCliffordPrefix::identity(n);
    for (g, t) in gates {
        prefix.apply_state_gate(g, t).unwrap();
    }
    for q in 0..n {
        let twisted_z = prefix.conjugate_z(q);
        let twisted_x = prefix.conjugate_x(q);
        let num_words = n.div_ceil(64).max(1);
        let mut z_only = SignedPauli::zero(num_words);
        z_only.set_z(q, true);
        let mut x_only = SignedPauli::zero(num_words);
        x_only.set_x(q, true);
        let lhs_z = pauli_string_expectation(&post, n, &z_only);
        let rhs_z = pauli_string_expectation(&pre, n, &twisted_z);
        assert!(
            (lhs_z - rhs_z).norm() < 1e-9,
            "Z_{q} mismatch: post={lhs_z} expected_from_prefix={rhs_z} (gates={gates:?})"
        );
        let lhs_x = pauli_string_expectation(&post, n, &x_only);
        let rhs_x = pauli_string_expectation(&pre, n, &twisted_x);
        assert!(
            (lhs_x - rhs_x).norm() < 1e-9,
            "X_{q} mismatch: post={lhs_x} expected_from_prefix={rhs_x} (gates={gates:?})"
        );
    }
}

#[test]
fn identity_is_identity() {
    let p = SignedCliffordPrefix::identity(3);
    for q in 0..3 {
        assert_eq!(p.conjugate_z(q).pauli_at(q), PauliKind::Z);
        assert_eq!(p.conjugate_z(q).phase4, 0);
        assert_eq!(p.conjugate_x(q).pauli_at(q), PauliKind::X);
        assert_eq!(p.conjugate_x(q).phase4, 0);
    }
}

#[test]
fn h_swaps_x_z() {
    assert_consistent(2, &[(Gate::H, vec![0])], 42);
    assert_consistent(2, &[(Gate::H, vec![1])], 42);
}

#[test]
fn s_and_sdg_signs() {
    assert_consistent(2, &[(Gate::S, vec![0])], 42);
    assert_consistent(2, &[(Gate::Sdg, vec![0])], 42);
    assert_consistent(2, &[(Gate::S, vec![0]), (Gate::S, vec![0])], 42);
}

#[test]
fn sx_and_sxdg() {
    assert_consistent(2, &[(Gate::SX, vec![0])], 42);
    assert_consistent(2, &[(Gate::SXdg, vec![0])], 42);
}

#[test]
fn pauli_gates_only_flip_signs() {
    assert_consistent(2, &[(Gate::X, vec![0])], 42);
    assert_consistent(2, &[(Gate::Y, vec![0])], 42);
    assert_consistent(2, &[(Gate::Z, vec![0])], 42);
}

#[test]
fn cx_and_cz() {
    assert_consistent(3, &[(Gate::Cx, vec![0, 1])], 42);
    assert_consistent(3, &[(Gate::Cz, vec![1, 2])], 42);
}

#[test]
fn swap_permutes_columns() {
    assert_consistent(3, &[(Gate::Swap, vec![0, 2])], 42);
}

#[test]
fn compound_h_cx() {
    assert_consistent(3, &[(Gate::H, vec![0]), (Gate::Cx, vec![0, 1])], 7);
}

#[test]
fn compound_h_cx_s() {
    assert_consistent(
        3,
        &[
            (Gate::H, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::S, vec![1]),
        ],
        7,
    );
}

#[test]
fn compound_h_cx_s_cz() {
    assert_consistent(
        3,
        &[
            (Gate::H, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::S, vec![1]),
            (Gate::Cz, vec![1, 2]),
        ],
        7,
    );
}

#[test]
fn random_clifford_sequence_4q() {
    let seq = vec![
        (Gate::H, vec![0]),
        (Gate::Cx, vec![0, 1]),
        (Gate::S, vec![1]),
        (Gate::Cz, vec![1, 2]),
        (Gate::H, vec![3]),
        (Gate::Cx, vec![2, 3]),
        (Gate::Sdg, vec![2]),
        (Gate::SX, vec![3]),
        (Gate::Z, vec![0]),
        (Gate::Swap, vec![0, 3]),
        (Gate::Y, vec![2]),
        (Gate::SXdg, vec![1]),
    ];
    assert_consistent(4, &seq, 17);
    assert_consistent(4, &seq, 31);
}

fn make_pauli(n: usize, factors: &[(usize, PauliKind)]) -> SignedPauli {
    let nw = n.div_ceil(64).max(1);
    let mut p = SignedPauli::zero(nw);
    for &(q, kind) in factors {
        match kind {
            PauliKind::I => {}
            PauliKind::X => p.set_x(q, true),
            PauliKind::Y => {
                p.set_x(q, true);
                p.set_z(q, true);
            }
            PauliKind::Z => p.set_z(q, true),
        }
    }
    p
}

fn empty_mps(n: usize) -> crate::backend::mps::MpsBackend {
    let mut b = crate::backend::mps::MpsBackend::new(0, 64);
    b.init(n, 0).unwrap();
    b
}

#[test]
fn ofd_no_xy_returns_none() {
    let mps = empty_mps(3);
    let p = make_pauli(3, &[(0, PauliKind::Z), (1, PauliKind::Z)]);
    assert!(
        build_ofd_disentangler(&mps, &p, 3, 1e-10)
            .unwrap()
            .is_none()
    );
}

#[test]
fn ofd_x_only_no_other_support_returns_empty_cascade() {
    let mps = empty_mps(3);
    let p = make_pauli(3, &[(0, PauliKind::X)]);
    let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
    assert_eq!(n, 0);
    assert!(d.is_empty());
}

#[test]
fn ofd_x0_z1_builds_cz() {
    let mps = empty_mps(3);
    let p = make_pauli(3, &[(0, PauliKind::X), (1, PauliKind::Z)]);
    let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
    assert_eq!(n, 0);
    assert_eq!(d.len(), 1);
    assert!(matches!(d[0].0, Gate::Cz));
    assert_eq!(d[0].1, vec![0, 1]);
}

#[test]
fn ofd_x0_x1_builds_cx() {
    let mps = empty_mps(3);
    let p = make_pauli(3, &[(0, PauliKind::X), (1, PauliKind::X)]);
    let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
    assert_eq!(n, 0);
    assert_eq!(d.len(), 1);
    assert!(matches!(d[0].0, Gate::Cx));
    assert_eq!(d[0].1, vec![0, 1]);
}

#[test]
fn ofd_x0_y1_builds_cy_decomposition() {
    let mps = empty_mps(3);
    let p = make_pauli(3, &[(0, PauliKind::X), (1, PauliKind::Y)]);
    let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
    assert_eq!(n, 0);
    assert_eq!(d.len(), 3);
    assert!(matches!(d[0].0, Gate::Sdg));
    assert_eq!(d[0].1, vec![1]);
    assert!(matches!(d[1].0, Gate::Cx));
    assert_eq!(d[1].1, vec![0, 1]);
    assert!(matches!(d[2].0, Gate::S));
    assert_eq!(d[2].1, vec![1]);
}

#[test]
fn ofd_multi_target_x_z() {
    let mps = empty_mps(4);
    let p = make_pauli(
        4,
        &[
            (0, PauliKind::X),
            (1, PauliKind::X),
            (2, PauliKind::Z),
            (3, PauliKind::I),
        ],
    );
    let (n, d) = build_ofd_disentangler(&mps, &p, 4, 1e-10).unwrap().unwrap();
    assert_eq!(n, 1, "anchor should sit at the routing-cost minimum");
    assert_eq!(d.len(), 2);
    assert!(matches!(d[0].0, Gate::Cx) && d[0].1 == vec![1, 0]);
    assert!(matches!(d[1].0, Gate::Cz) && d[1].1 == vec![1, 2]);
}

#[test]
fn ofd_skips_qubit_not_in_zero_state() {
    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(3, 0).unwrap();
    mps.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: SmallVec::from_slice(&[0]),
    })
    .unwrap();
    let p = make_pauli(
        3,
        &[(0, PauliKind::X), (1, PauliKind::Y), (2, PauliKind::Z)],
    );
    let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
    assert_eq!(n, 1);
    assert_eq!(d.len(), 2);
    assert!(matches!(d[0].0, Gate::Cx) && d[0].1 == vec![1, 0]);
    assert!(matches!(d[1].0, Gate::Cz) && d[1].1 == vec![1, 2]);
}

#[test]
fn ofd_post_apply_target_qubit_is_disentangled_zero() {
    let mut mps = empty_mps(3);
    mps.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[1]),
    })
    .unwrap();
    mps.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: SmallVec::from_slice(&[1, 2]),
    })
    .unwrap();
    let p = make_pauli(
        3,
        &[(0, PauliKind::X), (1, PauliKind::X), (2, PauliKind::X)],
    );
    let (n, d) = build_ofd_disentangler(&mps, &p, 3, 1e-10).unwrap().unwrap();
    assert_eq!(n, 0);
    for (gate, targets) in &d {
        mps.apply(&Instruction::Gate {
            gate: gate.clone(),
            targets: SmallVec::from_slice(targets),
        })
        .unwrap();
    }
    assert!(mps.is_qubit_in_zero_state(0, 1e-10).unwrap());
}

// apply_t_via_camps end-to-end tests

fn direct_state(n: usize, gates: &[(Gate, Vec<usize>)]) -> Vec<Complex64> {
    let mut sb = StatevectorBackend::new(0);
    sb.init(n, 0).unwrap();
    for (gate, targets) in gates {
        sb.apply(&Instruction::Gate {
            gate: gate.clone(),
            targets: SmallVec::from_slice(targets),
        })
        .unwrap();
    }
    sb.export_statevector().unwrap()
}

fn run_camps_then_t(
    n: usize,
    prep: &[(Gate, Vec<usize>)],
    t_target: usize,
    t_dagger: bool,
) -> (
    SignedCliffordPrefix,
    Vec<(Gate, Vec<usize>)>,
    crate::backend::mps::MpsBackend,
) {
    let mut prefix = SignedCliffordPrefix::identity(n);
    let mut prefix_gates: Vec<(Gate, Vec<usize>)> = Vec::new();
    for (gate, targets) in prep {
        prefix.apply_state_gate(gate, targets).unwrap();
        prefix_gates.push((gate.clone(), targets.clone()));
    }
    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    apply_t_via_camps(&mut prefix, &mut mps, t_target, t_dagger, 1e-10).unwrap();
    // After call, prefix has been updated to C·D† via fold_right_state_gate.
    // The tracker doesn't store gates, so callers that need to replay
    // the prefix onto a materialized state compare via Pauli expectations
    // instead of gate-list reconstruction.
    (prefix, prefix_gates, mps)
}

#[test]
fn t_via_camps_identity_prefix_falls_back_to_direct_t() {
    // C = I, so Z̄ = Z_target. OFD finds no disentangler, so the
    // pure-Z fallback applies T directly on the target qubit.
    let n = 2;
    let mut prefix = SignedCliffordPrefix::identity(n);
    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();

    let direct = direct_state(n, &[(Gate::T, vec![0])]);
    for q in 0..n {
        let zbar = prefix.conjugate_z(q);
        let factors = zbar.mps_factors(n);
        let zc = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!("non-real Z̄ phase {}", zbar.phase4),
        };
        let z_camps = phase_sign * zc.re;
        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();
        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "Z_{q}: camps={z_camps} direct={z_direct}"
        );
    }
}

#[test]
fn t_via_camps_with_h_prefix_matches_direct() {
    // Prep: H on qubit 0. Then T on qubit 0.
    // Direct: H, T on statevector.
    // CAMPS: prefix absorbs H, then apply_t_via_camps(0, T).
    let n = 2;
    let prep = vec![(Gate::H, vec![0])];
    let (prefix, _, mps) = run_camps_then_t(n, &prep, 0, false);

    // Prefix unitary is now H·... folded right by D†, giving C·D†.
    // Reconstructing C from the inverse tableau is impractical, so
    // this test compares via Pauli expectations instead of replay.
    let direct = direct_state(n, &[(Gate::H, vec![0]), (Gate::T, vec![0])]);

    // CAMPS oracle: ⟨ψ|Z_0|ψ⟩ = ⟨ϕ|C†Z_0C|ϕ⟩ = ⟨ϕ|Z̄'|ϕ⟩
    // where Z̄' = (new_C)†·Z_0·(new_C). Use prefix.conjugate_z(0)
    // and evaluate via mps.pauli_expectation.
    let zbar = prefix.conjugate_z(0);
    let factors = zbar.mps_factors(n);
    let z0_camps = mps.pauli_expectation(&factors).unwrap();
    let phase_sign = match zbar.phase4 & 3 {
        0 => 1.0,
        2 => -1.0,
        _ => panic!(
            "expected Hermitian Z̄ with phase4∈{{0,2}}, got {}",
            zbar.phase4
        ),
    };
    let z0_camps_real = phase_sign * z0_camps.re;

    let z0_direct: f64 = direct
        .iter()
        .enumerate()
        .map(|(i, amp)| {
            let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
            sign * amp.norm_sqr()
        })
        .sum();

    assert!(
        (z0_camps_real - z0_direct).abs() < 1e-9,
        "Z_0: camps={z0_camps_real} direct={z0_direct}"
    );
}

#[test]
fn t_via_camps_single_y_twisted_pauli_matches_direct() {
    let n = 1;
    let mut prefix = SignedCliffordPrefix::identity(n);
    prefix.apply_state_gate(&Gate::SX, &[0]).unwrap();

    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
    prefix.apply_state_gate(&Gate::H, &[0]).unwrap();

    let z_camps = eval_z(&prefix, &mps, &[0]);
    let direct = direct_state(
        n,
        &[(Gate::SX, vec![0]), (Gate::T, vec![0]), (Gate::H, vec![0])],
    );
    let z_direct: f64 = direct
        .iter()
        .enumerate()
        .map(|(i, amp)| {
            let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
            sign * amp.norm_sqr()
        })
        .sum();

    assert!(
        (z_camps - z_direct).abs() < 1e-9,
        "single-Y twisted Pauli: camps={z_camps} direct={z_direct}"
    );
}

#[test]
fn general_pauli_observable_matches_statevector() {
    use crate::sim::unified_pauli::{PauliAxis, PauliTerm};
    let n = 3;
    let prep = [
        (Gate::H, vec![0]),
        (Gate::SX, vec![1]),
        (Gate::Cx, vec![0, 1]),
        (Gate::S, vec![1]),
        (Gate::Cz, vec![1, 2]),
        (Gate::SXdg, vec![2]),
    ];
    let mut prefix = SignedCliffordPrefix::identity(n);
    for (g, t) in &prep {
        prefix.apply_state_gate(g, t).unwrap();
    }
    let mut mps = crate::backend::mps::MpsBackend::new(42, 64);
    mps.init(n, 0).unwrap();
    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();

    let mut full_gates = prep.to_vec();
    full_gates.push((Gate::T, vec![0]));
    let direct = direct_state(n, &full_gates);

    let cases: Vec<Vec<PauliTerm>> = vec![
        vec![PauliTerm::x(0)],
        vec![PauliTerm::y(1)],
        vec![PauliTerm::z(2), PauliTerm::x(0)],
        vec![PauliTerm::y(0), PauliTerm::y(2)],
        vec![PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)],
        vec![PauliTerm::y(0), PauliTerm::y(1), PauliTerm::y(2)],
    ];
    let num_words = n.div_ceil(64).max(1);
    for terms in &cases {
        let camps = evaluate_pauli_observable_camps(&prefix, &mps, terms).unwrap();
        let mut p = SignedPauli::zero(num_words);
        for t in terms {
            match t.axis {
                PauliAxis::X => p.set_x(t.qubit, true),
                PauliAxis::Z => p.set_z(t.qubit, true),
                PauliAxis::Y => {
                    p.set_x(t.qubit, true);
                    p.set_z(t.qubit, true);
                }
            }
        }
        let direct_val = pauli_string_expectation(&direct, n, &p).re;
        assert!(
            (camps - direct_val).abs() < 1e-9,
            "general Pauli {terms:?}: camps={camps} direct={direct_val}"
        );
    }
}

#[test]
fn t_via_camps_h_cx_prefix_multi_qubit_pauli() {
    // Prep: H_0, CX(0,1). Then T on qubit 0.
    // Z̄_0 = (H_0 CX_01)† Z_0 (H_0 CX_01) = H_0 Z_0 H_0 ⊗ ... etc.
    // Actually: (CX)†(H†) Z_0 (H)(CX). H Z_0 H = X_0. CX X_0 CX = X_0 X_1.
    // So Z̄ = X_0 X_1, letter at 0 is X, qubit 0 in |0⟩ (yes, fresh init), OFD succeeds.
    let n = 2;
    let prep = vec![(Gate::H, vec![0]), (Gate::Cx, vec![0, 1])];
    let (prefix, _, mps) = run_camps_then_t(n, &prep, 0, false);

    let direct = direct_state(
        n,
        &[
            (Gate::H, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::T, vec![0]),
        ],
    );

    for q in 0..n {
        let zbar = prefix.conjugate_z(q);
        let factors = zbar.mps_factors(n);
        let zc = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!("non-real Z̄ phase {}", zbar.phase4),
        };
        let z_camps = phase_sign * zc.re;

        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();

        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "Z_{q}: camps={z_camps} direct={z_direct}"
        );
    }
}

// deeper multi-qubit OFD success-path coverage

#[test]
fn t_via_camps_ofd_3q_mixed_cascade_matches_direct() {
    // Prefix: H_0, H_2, CX(0,1), CX(2,1), S_1; T on qubit 1.
    //   C = S_1 · CX(2,1) · CX(0,1) · H_2 · H_0
    //   C† Z_1 C: S_1† Z_1 S_1 = Z_1; CX(2,1) → Z_1 Z_2;
    //   CX(0,1) → Z_0 Z_1 Z_2; H_2 → Z_0 Z_1 X_2; H_0 → X_0 Z_1 X_2.
    // Mixed 3-qubit letters. OFD anchors on the first X/Y letter (q=0,
    // a valid |0⟩ anchor on the fresh MPS) and emits CZ(0,1), CX(0,2).
    let n = 3;
    let mut prefix = SignedCliffordPrefix::identity(n);
    for (g, t) in [
        (Gate::H, vec![0]),
        (Gate::H, vec![2]),
        (Gate::Cx, vec![0, 1]),
        (Gate::Cx, vec![2, 1]),
        (Gate::S, vec![1]),
    ] {
        prefix.apply_state_gate(&g, &t).unwrap();
    }

    let zbar_pre = prefix.conjugate_z(1);
    assert_eq!(zbar_pre.pauli_at(0), PauliKind::X);
    assert_eq!(zbar_pre.pauli_at(1), PauliKind::Z);
    assert_eq!(zbar_pre.pauli_at(2), PauliKind::X);

    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    // Verify OFD succeeds (fresh MPS, all qubits in |0⟩, X letter at qubit 0).
    let (anchor, cascade) = build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
        .unwrap()
        .unwrap();
    assert_eq!(anchor, 0);
    assert_eq!(cascade.len(), 2);
    assert!(matches!(cascade[0].0, Gate::Cz) && cascade[0].1 == vec![0, 1]);
    assert!(matches!(cascade[1].0, Gate::Cx) && cascade[1].1 == vec![0, 2]);

    apply_t_via_camps(&mut prefix, &mut mps, 1, false, 1e-10).unwrap();

    let direct = direct_state(
        n,
        &[
            (Gate::H, vec![0]),
            (Gate::H, vec![2]),
            (Gate::Cx, vec![0, 1]),
            (Gate::Cx, vec![2, 1]),
            (Gate::S, vec![1]),
            (Gate::T, vec![1]),
        ],
    );

    for q in 0..n {
        let zbar = prefix.conjugate_z(q);
        let factors = zbar.mps_factors(n);
        let zc = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!("non-real Z̄ phase {}", zbar.phase4),
        };
        let z_camps = phase_sign * zc.re;
        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();
        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "OFD-3q Z_{q}: camps={z_camps} direct={z_direct}"
        );
    }
}

#[test]
fn t_via_camps_ofd_with_y_letter_triplet_cascade() {
    // Force a Y letter to land in the twisted Pauli so OFD emits
    // the (Sdg, CX, S) Y-decomposition triplet.
    //
    // Prefix: CX(0,1), H_1, CX(1,2), S_2, H_2; T on qubit 2.
    //   C = H_2 · S_2 · CX(1,2) · H_1 · CX(0,1)
    //   C† Z_2 C: H_2 Z_2 H_2 = X_2; S_2† X_2 S_2 = -Y_2;
    //   CX(1,2) → -Z_1 Y_2; H_1 → -X_1 Y_2; CX(0,1) → -X_1 Y_2.
    // Z̄ = -X_1 Y_2, phase4 = 2, letters (I, X, Y). OFD anchors at q=1
    // (X, valid |0⟩ anchor); the Y at q=2 emits the (Sdg, CX, S) triplet.
    let n = 3;
    let mut prefix = SignedCliffordPrefix::identity(n);
    for (g, t) in [
        (Gate::Cx, vec![0, 1]),
        (Gate::H, vec![1]),
        (Gate::Cx, vec![1, 2]),
        (Gate::S, vec![2]),
        (Gate::H, vec![2]),
    ] {
        prefix.apply_state_gate(&g, &t).unwrap();
    }

    let zbar_pre = prefix.conjugate_z(2);
    assert_eq!(zbar_pre.pauli_at(0), PauliKind::I);
    assert_eq!(zbar_pre.pauli_at(1), PauliKind::X);
    assert_eq!(zbar_pre.pauli_at(2), PauliKind::Y);
    assert_eq!(zbar_pre.phase4, 2);

    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    let (anchor, cascade) = build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
        .unwrap()
        .unwrap();
    assert_eq!(anchor, 1);
    assert_eq!(
        cascade.len(),
        3,
        "Y partner should emit (Sdg, CX, S) triplet"
    );
    assert!(matches!(cascade[0].0, Gate::Sdg) && cascade[0].1 == vec![2]);
    assert!(matches!(cascade[1].0, Gate::Cx) && cascade[1].1 == vec![1, 2]);
    assert!(matches!(cascade[2].0, Gate::S) && cascade[2].1 == vec![2]);

    apply_t_via_camps(&mut prefix, &mut mps, 2, false, 1e-10).unwrap();

    let direct = direct_state(
        n,
        &[
            (Gate::Cx, vec![0, 1]),
            (Gate::H, vec![1]),
            (Gate::Cx, vec![1, 2]),
            (Gate::S, vec![2]),
            (Gate::H, vec![2]),
            (Gate::T, vec![2]),
        ],
    );

    for q in 0..n {
        let zbar = prefix.conjugate_z(q);
        let factors = zbar.mps_factors(n);
        let zc = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!("non-real Z̄ phase {}", zbar.phase4),
        };
        let z_camps = phase_sign * zc.re;
        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();
        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "OFD-Y-triplet Z_{q}: camps={z_camps} direct={z_direct}"
        );
    }
}

// OFDS tests

#[test]
fn ofds_x_anchor_no_zero_state_requirement() {
    let mut mps = empty_mps(2);
    mps.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: SmallVec::from_slice(&[0]),
    })
    .unwrap();
    mps.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: SmallVec::from_slice(&[1]),
    })
    .unwrap();
    let p = make_pauli(2, &[(0, PauliKind::X), (1, PauliKind::Z)]);
    assert!(
        build_ofd_disentangler(&mps, &p, 2, 1e-10)
            .unwrap()
            .is_none()
    );
    let (n, d) = build_ofds_disentangler(&mps, &p, 2).unwrap();
    assert_eq!(n, 0);
    assert_eq!(d.len(), 1);
    assert!(matches!(d[0].0, Gate::Cz) && d[0].1 == vec![0, 1]);
}

fn fresh_mps(n: usize) -> crate::backend::mps::MpsBackend {
    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    mps
}

#[test]
fn ofds_all_z_anchor_minimizes_routing() {
    let p = make_pauli(
        4,
        &[
            (0, PauliKind::Z),
            (1, PauliKind::I),
            (2, PauliKind::Z),
            (3, PauliKind::Z),
        ],
    );
    let mps = fresh_mps(4);
    let (n, d) = build_ofds_disentangler(&mps, &p, 4).unwrap();
    assert_eq!(n, 2, "anchor should sit at the routing-cost minimum");
    assert_eq!(d.len(), 2);
    assert!(matches!(d[0].0, Gate::Cx) && d[0].1 == vec![0, 2]);
    assert!(matches!(d[1].0, Gate::Cx) && d[1].1 == vec![3, 2]);
}

#[test]
fn ofds_single_z_returns_none() {
    let p = make_pauli(3, &[(1, PauliKind::Z)]);
    let mps = fresh_mps(3);
    assert!(build_ofds_disentangler(&mps, &p, 3).is_none());
}

#[test]
fn ofds_empty_returns_none() {
    let p = SignedPauli::zero(1);
    let mps = fresh_mps(3);
    assert!(build_ofds_disentangler(&mps, &p, 3).is_none());
}

#[test]
fn t_via_camps_ofds_xy_path_matches_direct() {
    let n = 2;
    let mut prefix = SignedCliffordPrefix::identity(n);
    prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
    prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();

    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    mps.apply(&Instruction::Gate {
        gate: Gate::Ry(0.7),
        targets: SmallVec::from_slice(&[0]),
    })
    .unwrap();
    mps.apply(&Instruction::Gate {
        gate: Gate::Ry(0.5),
        targets: SmallVec::from_slice(&[1]),
    })
    .unwrap();

    let zbar_pre = prefix.conjugate_z(0);
    assert!(
        build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
            .unwrap()
            .is_none(),
        "test setup error: OFD should have failed so OFDS path is exercised"
    );

    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();

    let direct = direct_state(
        n,
        &[
            (Gate::Ry(0.7), vec![0]),
            (Gate::Ry(0.5), vec![1]),
            (Gate::H, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::T, vec![0]),
        ],
    );

    for q in 0..n {
        let zbar = prefix.conjugate_z(q);
        let factors = zbar.mps_factors(n);
        let zc = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!("non-real Z̄ phase {}", zbar.phase4),
        };
        let z_camps = phase_sign * zc.re;
        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();
        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "OFDS Z_{q}: camps={z_camps} direct={z_direct}"
        );
    }
}

#[test]
fn t_via_camps_ofds_all_z_path_matches_direct() {
    let n = 3;
    let mut prefix = SignedCliffordPrefix::identity(n);
    prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
    prefix.apply_state_gate(&Gate::Cx, &[1, 2]).unwrap();

    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();

    let zbar_pre = prefix.conjugate_z(2);
    for q in 0..n {
        assert!(
            matches!(zbar_pre.pauli_at(q), PauliKind::Z),
            "test setup: expected all-Z twisted Pauli, got {:?} at q={q}",
            zbar_pre.pauli_at(q)
        );
    }
    assert!(
        build_ofd_disentangler(&mps, &zbar_pre, n, 1e-10)
            .unwrap()
            .is_none(),
        "test setup: OFD should reject all-Z support"
    );

    apply_t_via_camps(&mut prefix, &mut mps, 2, false, 1e-10).unwrap();

    let direct = direct_state(
        n,
        &[
            (Gate::Cx, vec![0, 1]),
            (Gate::Cx, vec![1, 2]),
            (Gate::T, vec![2]),
        ],
    );

    for q in 0..n {
        let zbar = prefix.conjugate_z(q);
        let factors = zbar.mps_factors(n);
        let zc = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!("non-real Z̄ phase {}", zbar.phase4),
        };
        let z_camps = phase_sign * zc.re;
        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();
        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "OFDS all-Z Z_{q}: camps={z_camps} direct={z_direct}"
        );
    }
}

#[test]
fn tdag_via_camps_h_prefix() {
    let n = 2;
    let prep = vec![(Gate::H, vec![0])];
    let (prefix, _, mps) = run_camps_then_t(n, &prep, 0, true);

    let direct = direct_state(n, &[(Gate::H, vec![0]), (Gate::Tdg, vec![0])]);

    for q in 0..n {
        let zbar = prefix.conjugate_z(q);
        let factors = zbar.mps_factors(n);
        let zc = mps.pauli_expectation(&factors).unwrap();
        let phase_sign = match zbar.phase4 & 3 {
            0 => 1.0,
            2 => -1.0,
            _ => panic!("non-real Z̄ phase {}", zbar.phase4),
        };
        let z_camps = phase_sign * zc.re;
        let z_direct: f64 = direct
            .iter()
            .enumerate()
            .map(|(i, amp)| {
                let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                sign * amp.norm_sqr()
            })
            .sum();
        assert!(
            (z_camps - z_direct).abs() < 1e-9,
            "Tdg Z_{q}: camps={z_camps} direct={z_direct}"
        );
    }
}

fn two_t_zz_probe(post_cliffords: &[(Gate, Vec<usize>)]) -> (f64, f64) {
    let n = 2;
    let mut prefix = SignedCliffordPrefix::identity(n);
    prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
    prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
    prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
    for (g, t) in post_cliffords {
        prefix.apply_state_gate(g, t).unwrap();
    }
    let zz_camps = eval_z(&prefix, &mps, &[0, 1]);
    let mut gates: Vec<(Gate, Vec<usize>)> = vec![
        (Gate::H, vec![0]),
        (Gate::Cx, vec![0, 1]),
        (Gate::T, vec![0]),
        (Gate::H, vec![0]),
        (Gate::T, vec![0]),
    ];
    gates.extend_from_slice(post_cliffords);
    let direct = direct_state(n, &gates);
    let zz_direct: f64 = direct
        .iter()
        .enumerate()
        .map(|(i, amp)| {
            let s0 = if i & 1 == 0 { 1.0 } else { -1.0 };
            let s1 = if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 };
            s0 * s1 * amp.norm_sqr()
        })
        .sum();
    (zz_camps, zz_direct)
}

#[test]
fn t_via_camps_two_t_bisect_post_cliffords() {
    type BisectCase<'a> = (&'a str, &'a [(Gate, &'a [usize])]);
    let cases: &[BisectCase] = &[
        ("no_post", &[]),
        ("h0", &[(Gate::H, &[0])]),
        ("h1", &[(Gate::H, &[1])]),
        ("h0_h1", &[(Gate::H, &[0]), (Gate::H, &[1])]),
        ("h1_h0", &[(Gate::H, &[1]), (Gate::H, &[0])]),
        ("x0_x1", &[(Gate::X, &[0]), (Gate::X, &[1])]),
        ("z0_z1", &[(Gate::Z, &[0]), (Gate::Z, &[1])]),
        ("cx_01", &[(Gate::Cx, &[0, 1])]),
        ("s0_s1", &[(Gate::S, &[0]), (Gate::S, &[1])]),
    ];
    let mut failures = Vec::new();
    for (label, ops) in cases {
        let owned: Vec<(Gate, Vec<usize>)> =
            ops.iter().map(|(g, t)| (g.clone(), t.to_vec())).collect();
        let (c, d) = two_t_zz_probe(&owned);
        if (c - d).abs() >= 1e-9 {
            failures.push(format!("{label}: camps={c} direct={d}"));
        }
    }
    assert!(failures.is_empty(), "fails:\n  {}", failures.join("\n  "));
}

#[test]
fn t_via_camps_two_t_multi_qubit_zz_observable() {
    // Reproduces the prior multi-qubit two-T failure shape on a small fixture
    // and checks the joint ⟨Z_0 Z_1⟩ against statevector.
    // Sequence: H_0; CX(0,1); T_0; H_0; T_0
    let n = 2;
    let mut prefix = SignedCliffordPrefix::identity(n);
    prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
    prefix.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();

    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();

    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
    prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
    apply_t_via_camps(&mut prefix, &mut mps, 0, false, 1e-10).unwrap();
    prefix.apply_state_gate(&Gate::H, &[0]).unwrap();
    prefix.apply_state_gate(&Gate::H, &[1]).unwrap();

    let zz_camps = eval_z(&prefix, &mps, &[0, 1]);

    let direct = direct_state(
        n,
        &[
            (Gate::H, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::T, vec![0]),
            (Gate::H, vec![0]),
            (Gate::T, vec![0]),
            (Gate::H, vec![0]),
            (Gate::H, vec![1]),
        ],
    );
    let zz_direct: f64 = direct
        .iter()
        .enumerate()
        .map(|(i, amp)| {
            let s0 = if i & 1 == 0 { 1.0 } else { -1.0 };
            let s1 = if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 };
            s0 * s1 * amp.norm_sqr()
        })
        .sum();

    assert!(
        (zz_camps - zz_direct).abs() < 1e-9,
        "two-T ⟨Z_0 Z_1⟩: camps={zz_camps} direct={zz_direct}"
    );
}

#[test]
fn t_via_camps_bond_dim_stays_bounded_on_ofd_and_ofds_paths() {
    let max_bond = 64;

    let n_ofd = 3;
    let mut prefix_ofd = SignedCliffordPrefix::identity(n_ofd);
    prefix_ofd.apply_state_gate(&Gate::H, &[0]).unwrap();
    prefix_ofd.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
    prefix_ofd.apply_state_gate(&Gate::Cx, &[1, 2]).unwrap();
    let mut mps_ofd = crate::backend::mps::MpsBackend::new(0, max_bond);
    mps_ofd.init(n_ofd, 0).unwrap();

    let zbar_ofd = prefix_ofd.conjugate_z(0);
    assert!(
        build_ofd_disentangler(&mps_ofd, &zbar_ofd, n_ofd, 1e-10)
            .unwrap()
            .is_some(),
        "test setup: OFD should succeed for this prefix"
    );

    apply_t_via_camps(&mut prefix_ofd, &mut mps_ofd, 0, false, 1e-10).unwrap();
    let ofd_peak = mps_ofd.current_max_bond_dim();
    assert!(
        ofd_peak <= max_bond,
        "OFD path exceeded max_bond_dim cap: peak={ofd_peak} cap={max_bond}"
    );

    let n_ofds = 3;
    let mut prefix_ofds = SignedCliffordPrefix::identity(n_ofds);
    prefix_ofds.apply_state_gate(&Gate::Cx, &[0, 1]).unwrap();
    prefix_ofds.apply_state_gate(&Gate::Cx, &[1, 2]).unwrap();
    let mut mps_ofds = crate::backend::mps::MpsBackend::new(0, max_bond);
    mps_ofds.init(n_ofds, 0).unwrap();

    let zbar_ofds = prefix_ofds.conjugate_z(2);
    assert!(
        build_ofd_disentangler(&mps_ofds, &zbar_ofds, n_ofds, 1e-10)
            .unwrap()
            .is_none(),
        "test setup: OFD should reject all-Z support so OFDS is exercised"
    );

    apply_t_via_camps(&mut prefix_ofds, &mut mps_ofds, 2, false, 1e-10).unwrap();
    let ofds_peak = mps_ofds.current_max_bond_dim();
    assert!(
        ofds_peak <= max_bond,
        "OFDS path exceeded max_bond_dim cap: peak={ofds_peak} cap={max_bond}"
    );
}

#[test]
fn t_via_camps_ofds_saturation_guard_errors_when_cap_too_small() {
    let n = 4;
    let mut prefix = SignedCliffordPrefix::identity(n);
    for q in 0..n - 1 {
        prefix.apply_state_gate(&Gate::Cx, &[q, q + 1]).unwrap();
    }

    // Mixed product state: even qubits in |+>, odd qubits in |0>. The
    // all-Z OFDS CX ladder then has |+>-control / |0>-target rungs, which
    // create Bell pairs the bond-dim-1 cap must truncate.
    let mut mps = crate::backend::mps::MpsBackend::new(0, 1);
    mps.init(n, 0).unwrap();
    for q in (0..n).step_by(2) {
        mps.apply(&crate::circuit::Instruction::Gate {
            gate: Gate::H,
            targets: crate::circuit::SmallVec::from_slice(&[q]),
        })
        .unwrap();
    }

    let zbar = prefix.conjugate_z(n - 1);
    assert!(
        build_ofd_disentangler(&mps, &zbar, n, 1e-10)
            .unwrap()
            .is_none(),
        "test setup: OFD must decline so OFDS path runs"
    );

    let result = apply_t_via_camps(&mut prefix, &mut mps, n - 1, false, 1e-10);
    let err = result.expect_err("OFDS truncation at bond cap=1 should trip the guard");
    let msg = format!("{err}");
    assert!(
        msg.contains("SVD truncation discarded"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn choose_disentangler_prefers_ofds_when_routing_cost_strictly_lower() {
    let n = 5;
    let mut mps = fresh_mps(n);
    mps.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[2]),
    })
    .unwrap();
    let p = make_pauli(
        n,
        &[(0, PauliKind::X), (2, PauliKind::X), (4, PauliKind::X)],
    );

    let (ofd_anchor, ofd_cascade) = build_ofd_disentangler(&mps, &p, n, 1e-10)
        .unwrap()
        .expect("OFD has |0⟩ X anchors at q0 and q4");
    assert!(
        ofd_anchor == 0 || ofd_anchor == 4,
        "OFD must reject q2 because it is in |+⟩, picked {ofd_anchor}"
    );
    let ofd_cost = cascade_routing_cost(&mps, &ofd_cascade);
    assert_eq!(ofd_cost, 6, "OFD anchor at q0/q4 routes |0-2|+|0-4|=6");

    let (ofds_anchor, ofds_cascade) =
        build_ofds_disentangler(&mps, &p, n).expect("OFDS has full X anchor freedom");
    assert_eq!(
        ofds_anchor, 2,
        "OFDS picks q2 (middle of support, no |0⟩ requirement)"
    );
    let ofds_cost = cascade_routing_cost(&mps, &ofds_cascade);
    assert_eq!(ofds_cost, 4, "OFDS anchor at q2 routes |2-0|+|2-4|=4");

    let (chosen_anchor, chosen_cascade, chosen_kind) = choose_disentangler(&mps, &p, n, 1e-10)
        .unwrap()
        .expect("cost-compare picks the cheaper cascade");
    assert_eq!(chosen_kind, DisentanglerKind::Ofds);
    assert_eq!(chosen_anchor, 2);
    assert_eq!(cascade_routing_cost(&mps, &chosen_cascade), 4);
}

#[test]
fn choose_disentangler_breaks_ties_in_favor_of_ofd_for_bond_dim_safety() {
    let n = 3;
    let mps = fresh_mps(n);
    let p = make_pauli(
        n,
        &[(0, PauliKind::X), (1, PauliKind::X), (2, PauliKind::X)],
    );

    let (_, ofd_cascade) = build_ofd_disentangler(&mps, &p, n, 1e-10)
        .unwrap()
        .expect("OFD finds anchor on |0...0⟩");
    let (_, ofds_cascade) = build_ofds_disentangler(&mps, &p, n).expect("OFDS also finds anchor");
    assert_eq!(
        cascade_routing_cost(&mps, &ofd_cascade),
        cascade_routing_cost(&mps, &ofds_cascade),
        "this fixture must produce a routing-cost tie"
    );

    let (_, _, chosen_kind) = choose_disentangler(&mps, &p, n, 1e-10)
        .unwrap()
        .expect("dispatch returns Some on a viable support");
    assert_eq!(
        chosen_kind,
        DisentanglerKind::Ofd,
        "ties must go to OFD because OFD preserves bond dimension"
    );
}

#[test]
fn apply_t_via_camps_with_cost_compare_dispatch_matches_statevector() {
    let n = 5;
    let mut prefix = SignedCliffordPrefix::identity(n);
    let mut mps = crate::backend::mps::MpsBackend::new(0, 64);
    mps.init(n, 0).unwrap();
    mps.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[2]),
    })
    .unwrap();

    let mut sv = StatevectorBackend::new(0);
    sv.init(n, 0).unwrap();
    sv.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[2]),
    })
    .unwrap();

    apply_t_via_camps(&mut prefix, &mut mps, 2, false, 1e-10).unwrap();
    sv.apply(&Instruction::Gate {
        gate: Gate::T,
        targets: SmallVec::from_slice(&[2]),
    })
    .unwrap();

    let probs = sv.probabilities().unwrap();
    for q in 0..n {
        let camps = eval_z(&prefix, &mps, &[q]);
        let mask = 1usize << q;
        let direct: f64 = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| if i & mask == 0 { p } else { -p })
            .sum();
        assert!(
            (camps - direct).abs() < 1e-9,
            "Z_{q}: camps={camps:.12}, statevector={direct:.12}"
        );
    }
}

#[test]
fn ofd_anchor_heuristic_picks_routing_minimum() {
    let n = 5;
    let mps = fresh_mps(n);
    let p = make_pauli(
        n,
        &[
            (0, PauliKind::X),
            (2, PauliKind::Z),
            (3, PauliKind::X),
            (4, PauliKind::Z),
        ],
    );
    let (anchor, _) = build_ofd_disentangler(&mps, &p, n, 1e-10)
        .unwrap()
        .expect("OFD should succeed on fresh |0...0⟩");
    let support = support_qubits(&p, n);
    let xy_candidates: Vec<usize> = support
        .iter()
        .copied()
        .filter(|&q| matches!(p.pauli_at(q), PauliKind::X | PauliKind::Y))
        .collect();
    let chosen_cost = anchor_routing_cost(&mps, anchor, &support);
    for &alt in &xy_candidates {
        let alt_cost = anchor_routing_cost(&mps, alt, &support);
        assert!(
            chosen_cost <= alt_cost,
            "heuristic picked anchor {anchor} (cost {chosen_cost}) but {alt} costs {alt_cost}"
        );
    }
}
