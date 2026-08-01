use super::*;
use crate::circuit::SmallVec;

#[test]
fn test_pure_clifford() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);

    let result = run_stabilizer_rank(&c, 42).unwrap();
    assert_eq!(result.num_terms, 1);
    assert_eq!(result.t_count, 0);
    assert!((result.probabilities[0] - 0.5).abs() < 1e-10);
    assert!((result.probabilities[3] - 0.5).abs() < 1e-10);
}

#[test]
fn test_single_t() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);

    let result = run_stabilizer_rank(&c, 42).unwrap();
    assert_eq!(result.num_terms, 2);
    assert_eq!(result.t_count, 1);

    let p0_expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
    assert!(
        (result.probabilities[0] - p0_expected).abs() < 1e-10,
        "P(0) = {}, expected {}",
        result.probabilities[0],
        p0_expected
    );
}

#[test]
fn shots_preserve_t_branch_interference() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);

    let num_shots = 20_000;
    let result = run_stabilizer_rank_shots(&c, num_shots, 42).unwrap();
    let zeros = result.shots.iter().filter(|s| !s[0]).count();
    let p0 = zeros as f64 / num_shots as f64;
    let expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
    assert!(
        (p0 - expected).abs() < 0.02,
        "P(0) = {p0}, expected {expected} (a classical T-branch mixture would give 0.5)"
    );
}

#[test]
fn shots_without_t_bypass_statevector_qubit_cap() {
    let n = MAX_STATEVECTOR_QUBITS + 5;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
        c.add_measure(q, q);
    }

    let result = run_stabilizer_rank_shots(&c, 16, 42).unwrap();
    assert_eq!(result.shots.len(), 16);
    assert!(result.shots.iter().all(|shot| shot.len() == n));
}

#[test]
fn shots_with_t_bypass_statevector_qubit_cap_terminal() {
    let n = MAX_STATEVECTOR_QUBITS + 5;
    let mut c = Circuit::new(n, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);

    let result = run_stabilizer_rank_shots(&c, 32, 42).unwrap();
    assert_eq!(result.shots.len(), 32);
    assert!(result.shots.iter().all(|shot| shot.len() == 1));

    let public_result =
        crate::sim::run_shots_with(crate::sim::BackendKind::StabilizerRank, &c, 8, 42).unwrap();
    assert_eq!(public_result.shots.len(), 8);

    let auto_result = crate::sim::run_shots_with(crate::sim::BackendKind::Auto, &c, 8, 42).unwrap();
    assert_eq!(auto_result.shots.len(), 8);
}

#[test]
fn shots_with_t_bypass_statevector_qubit_cap_mid_circuit() {
    let n = MAX_STATEVECTOR_QUBITS + 5;
    let mut c = Circuit::new(n, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[1]),
    });
    c.add_reset(0);
    c.add_measure(1, 1);

    let result = run_stabilizer_rank_shots(&c, 32, 42).unwrap();
    assert_eq!(result.shots.len(), 32);
    assert!(result.shots.iter().all(|shot| shot.len() == 2));
}

#[test]
fn forced_mps_projection_has_expected_probability() {
    let mut plus = MpsBackend::new_exact(0);
    plus.init(1, 0).unwrap();
    plus.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[0]),
    })
    .unwrap();

    let mut zero = plus.clone();
    let mut one = plus;
    let p0 = zero.project_z_outcome(0, false);
    let p1 = one.project_z_outcome(0, true);

    assert!((p0 - 0.5).abs() < 1e-12);
    assert!((p1 - 0.5).abs() < 1e-12);
    assert!(zero.inner_product(&one).unwrap().norm() < 1e-12);
}

#[test]
fn test_rejects_reset_in_probability_path() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_reset(0);

    assert!(run_stabilizer_rank(&c, 42).is_err());
    assert!(run_stabilizer_rank_approx(&c, 8, 42).is_err());
}

#[test]
fn test_rejects_measurement_in_probability_path() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_measure(0, 0);

    assert!(run_stabilizer_rank(&c, 42).is_err());
    assert!(run_stabilizer_rank_approx(&c, 8, 42).is_err());
}

#[test]
fn test_multi_t_with_separating_clifford() {
    // Regression: prior to absorbing the deterministic Z-eigenvalue
    // into the branch weight, this circuit returned [0.5, 0.5]
    // because the AG tableau dropped the global -1 phase on the
    // |1⟩ branch. The T-count scaling sweep surfaced the bug.
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);

    let result = run_stabilizer_rank(&c, 42).unwrap();
    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    for (i, (&sr, &sv)) in result.probabilities.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (sr - sv).abs() < 1e-10,
            "P({i}) mismatch: stab_rank = {sr}, statevector = {sv}"
        );
    }
}

// Regression for prior two-T multi-qubit reconstruction failure.
// Per-branch tableau export picked inconsistent implicit global
// phases when support shifted between branches. The Pauli-offset
// representation should preserve the interference pattern.
#[test]
fn test_two_t_multi_qubit_bisect_stages() {
    type Stage<'a> = (&'a str, &'a [(Gate, &'a [usize])]);
    let stages: &[Stage] = &[
        ("ghz_only", &[(Gate::H, &[0]), (Gate::Cx, &[0, 1])]),
        (
            "ghz_t",
            &[(Gate::H, &[0]), (Gate::Cx, &[0, 1]), (Gate::T, &[0])],
        ),
        (
            "ghz_t_h",
            &[
                (Gate::H, &[0]),
                (Gate::Cx, &[0, 1]),
                (Gate::T, &[0]),
                (Gate::H, &[0]),
            ],
        ),
        (
            "ghz_t_h_t",
            &[
                (Gate::H, &[0]),
                (Gate::Cx, &[0, 1]),
                (Gate::T, &[0]),
                (Gate::H, &[0]),
                (Gate::T, &[0]),
            ],
        ),
        (
            "ghz_t_h_t_h0",
            &[
                (Gate::H, &[0]),
                (Gate::Cx, &[0, 1]),
                (Gate::T, &[0]),
                (Gate::H, &[0]),
                (Gate::T, &[0]),
                (Gate::H, &[0]),
            ],
        ),
        (
            "ghz_t_h_t_h0_h1",
            &[
                (Gate::H, &[0]),
                (Gate::Cx, &[0, 1]),
                (Gate::T, &[0]),
                (Gate::H, &[0]),
                (Gate::T, &[0]),
                (Gate::H, &[0]),
                (Gate::H, &[1]),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for (label, gates) in stages {
        let mut c = Circuit::new(2, 0);
        for (gate, targets) in *gates {
            c.add_gate(gate.clone(), targets);
        }
        let result = run_stabilizer_rank(&c, 42).unwrap();
        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        let max_diff = result
            .probabilities
            .iter()
            .zip(sv_probs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        if max_diff > 1e-9 {
            failures.push(format!(
                "{label}: sr={:?} sv={:?}",
                result.probabilities, sv_probs
            ));
        }
    }
    assert!(failures.is_empty(), "fails:\n  {}", failures.join("\n  "));
}

// Companion to the bisect test: minimal multi-qubit two-T fixture.
// Same root cause: cross-branch phase reconstruction.
#[test]
fn test_two_t_multi_qubit_entangled_matches_statevector() {
    // Surface fixture: H_0, CX(0,1), T_0, H_0, T_0, H_0, H_1.
    // Both stabilizer_rank and statevector should agree on the
    // full 2q probability vector.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    let result = run_stabilizer_rank(&c, 42).unwrap();
    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    for (i, (&sr, &sv)) in result.probabilities.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (sr - sv).abs() < 1e-10,
            "P({i}) mismatch: stab_rank = {sr}, statevector = {sv}"
        );
    }
}

#[test]
fn test_multi_qubit_multi_t_post_cliffords_matches_statevector() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[0, 2]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_gate(Gate::Tdg, &[2]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Swap, &[0, 2]);

    let sr = run_stabilizer_rank(&c, 42).unwrap();
    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    for (i, (sr_p, sv_p)) in sr.probabilities.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (sr_p - sv_p).abs() < 1e-10,
            "prob[{i}]: stab_rank={sr_p}, statevector={sv_p}"
        );
    }
}

#[test]
fn test_matches_statevector() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[2, 1]);

    let sr = run_stabilizer_rank(&c, 42).unwrap();
    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();

    for (i, (sr_p, sv_p)) in sr.probabilities.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (sr_p - sv_p).abs() < 1e-10,
            "prob[{i}]: stab_rank={sr_p}, statevector={sv_p}"
        );
    }
}

#[test]
fn test_tdg() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Tdg, &[0]);
    c.add_gate(Gate::H, &[0]);

    let result = run_stabilizer_rank(&c, 42).unwrap();
    assert_eq!(result.t_count, 1);

    let p0_expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
    assert!((result.probabilities[0] - p0_expected).abs() < 1e-10);
}

#[test]
fn test_term_count_scaling() {
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }

    let result = run_stabilizer_rank(&c, 42).unwrap();
    assert_eq!(result.t_count, 4);
    assert_eq!(result.num_terms, 16); // 2^4

    let total: f64 = result.probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-8);
}

#[test]
fn test_overlap_identical_states() {
    let mut b1 = StabilizerBackend::new(42);
    b1.init(3, 0).unwrap();
    let inst_h = Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[0]),
    };
    let inst_cx = Instruction::Gate {
        gate: Gate::Cx,
        targets: SmallVec::from_slice(&[0, 1]),
    };
    b1.apply(&inst_h).unwrap();
    b1.apply(&inst_cx).unwrap();

    let b2 = b1.clone();
    let overlap = stabilizer_overlap_sq(&b1, &b2, 3);
    assert!(
        (overlap - 1.0).abs() < 1e-10,
        "overlap of identical states should be 1, got {}",
        overlap
    );
}

#[test]
fn test_overlap_orthogonal_states() {
    // |0⟩ and |1⟩ are orthogonal
    let mut b1 = StabilizerBackend::new(42);
    b1.init(1, 0).unwrap();
    // b1 = |0⟩

    let mut b2 = StabilizerBackend::new(42);
    b2.init(1, 0).unwrap();
    let inst_x = Instruction::Gate {
        gate: Gate::X,
        targets: SmallVec::from_slice(&[0]),
    };
    b2.apply(&inst_x).unwrap();
    // b2 = |1⟩

    let overlap = stabilizer_overlap_sq(&b1, &b2, 1);
    assert!(
        overlap < 1e-10,
        "overlap of |0⟩ and |1⟩ should be 0, got {}",
        overlap
    );
}

#[test]
fn test_overlap_bell_with_basis() {
    // |Φ+⟩ = (|00⟩+|11⟩)/√2 vs |00⟩: |⟨00|Φ+⟩|² = 1/2
    let mut bell = StabilizerBackend::new(42);
    bell.init(2, 0).unwrap();
    let inst_h = Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[0]),
    };
    let inst_cx = Instruction::Gate {
        gate: Gate::Cx,
        targets: SmallVec::from_slice(&[0, 1]),
    };
    bell.apply(&inst_h).unwrap();
    bell.apply(&inst_cx).unwrap();

    let mut basis = StabilizerBackend::new(42);
    basis.init(2, 0).unwrap();

    let overlap = stabilizer_overlap_sq(&bell, &basis, 2);
    assert!(
        (overlap - 0.5).abs() < 1e-10,
        "|⟨00|Φ+⟩|² should be 0.5, got {}",
        overlap
    );
}

#[test]
fn test_overlap_plus_with_basis() {
    // |+⟩ vs |0⟩: |⟨0|+⟩|² = 1/2
    let mut plus = StabilizerBackend::new(42);
    plus.init(1, 0).unwrap();
    let inst_h = Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[0]),
    };
    plus.apply(&inst_h).unwrap();

    let mut zero = StabilizerBackend::new(42);
    zero.init(1, 0).unwrap();

    let overlap = stabilizer_overlap_sq(&plus, &zero, 1);
    assert!(
        (overlap - 0.5).abs() < 1e-10,
        "|⟨0|+⟩|² should be 0.5, got {}",
        overlap
    );
}

#[test]
fn test_stabilizer_inner_product_matches_dense_export() {
    let mut b1 = StabilizerBackend::new(42);
    b1.init(2, 0).unwrap();
    b1.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[0]),
    })
    .unwrap();
    b1.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: SmallVec::from_slice(&[0, 1]),
    })
    .unwrap();

    let mut b2 = StabilizerBackend::new(7);
    b2.init(2, 0).unwrap();
    b2.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: SmallVec::from_slice(&[0]),
    })
    .unwrap();
    b2.apply(&Instruction::Gate {
        gate: Gate::S,
        targets: SmallVec::from_slice(&[0]),
    })
    .unwrap();
    b2.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: SmallVec::from_slice(&[0, 1]),
    })
    .unwrap();

    let b1_vec = b1.export_statevector().unwrap();
    let b2_vec = b2.export_statevector().unwrap();
    let expected: Complex64 = b1_vec
        .iter()
        .zip(b2_vec.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    let actual = stabilizer_inner_product(&b1, &b2, 2).unwrap();
    assert!((actual - expected).norm() < 1e-12);
}

#[test]
fn test_too_many_terms() {
    let mut c = Circuit::new(1, 0);
    // 21 T gates would need 2^21 > MAX_TERMS terms
    for _ in 0..21 {
        c.add_gate(Gate::T, &[0]);
    }
    let result = run_stabilizer_rank(&c, 42);
    assert!(result.is_err());
}

#[test]
fn test_approx_small_circuit_exact() {
    // With budget > 2^t, approximate = exact
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::T, &[1]);

    let exact = run_stabilizer_rank(&c, 42).unwrap();
    let approx = run_stabilizer_rank_approx(&c, 1024, 42).unwrap();

    assert_eq!(approx.num_terms, exact.num_terms);
    assert_eq!(approx.pruned_count, 0);
    for (e, a) in exact.probabilities.iter().zip(approx.probabilities.iter()) {
        assert!((e - a).abs() < 1e-10);
    }
}

#[test]
fn test_approx_prunes_terms() {
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    // 4 T gates → 16 terms exact. Budget of 8 should prune.
    let result = run_stabilizer_rank_approx(&c, 8, 42).unwrap();
    assert!(result.num_terms <= 8);
    assert!(result.pruned_count > 0);

    let total: f64 = result.probabilities.iter().sum();
    // Approximate, so not exactly 1.0, but should be in a reasonable range
    assert!(total > 0.5 && total < 2.0, "total = {total}");
}

#[test]
fn test_approx_handles_many_t_gates() {
    // 10 T gates → 1024 exact terms, budget 32 should work without error
    let mut c = Circuit::new(3, 0);
    for _ in 0..10 {
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
    }
    let result = run_stabilizer_rank_approx(&c, 32, 42).unwrap();
    assert!(result.num_terms <= 32);
    assert_eq!(result.t_count, 10);
}
