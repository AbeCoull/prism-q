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

fn eight_t_circuit() -> Circuit {
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    for q in 0..3 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..4 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    c
}

#[test]
fn pruned_runs_stay_inside_the_reported_error_bound() {
    let c = eight_t_circuit();
    let exact = run_stabilizer_rank(&c, 42).unwrap();
    assert_eq!(exact.num_terms, 256);

    for budget in [64usize, 160, 224] {
        let approx = run_stabilizer_rank_approx(&c, budget, 42).unwrap();
        assert!(approx.num_terms <= budget);
        assert!(approx.pruned_count > 0);

        let total: f64 = approx.probabilities.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "budget {budget} summed to {total}"
        );

        let deviation: f64 = exact
            .probabilities
            .iter()
            .zip(&approx.probabilities)
            .map(|(e, a)| (e - a).abs())
            .sum();
        let bound = 4.0 * approx.discarded_weight;
        assert!(
            deviation <= bound + 1e-9,
            "budget {budget} deviated {deviation} past its bound {bound}"
        );
    }

    let whole = run_stabilizer_rank_approx(&c, 256, 42).unwrap();
    assert_eq!(whole.pruned_count, 0);
    assert_eq!(whole.discarded_weight, 0.0);
}

// The 1-norm bound is worst case over constructive interference between
// non-orthogonal branches, so it certifies nothing at the budgets worth
// taking. Both budgets here straddle a class of equal-magnitude branches, so
// the assertions read the discarded magnitudes, which the tie-break cannot
// move, rather than the distribution, which it can.
#[test]
fn the_error_bound_certifies_only_a_generous_budget() {
    let c = eight_t_circuit();

    let tight = run_stabilizer_rank_approx(&c, 64, 42).unwrap();
    assert!(tight.pruned_count > 0);
    assert_eq!(tight.fidelity_bound(), 0.0);

    let generous = run_stabilizer_rank_approx(&c, 224, 42).unwrap();
    assert!(generous.pruned_count > 0);
    assert!(generous.fidelity_bound() > 0.97);
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

// Term (i, j) of the branch Gram is the conjugate of (j, i); the triangle
// evaluation must match the full B x B sum.
#[test]
fn triangle_gram_matches_the_full_sum() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::T, &[2]);

    let branches = build_mps_branches_for_unitary(&c, 42).unwrap();
    assert!(branches.len() >= 8, "expected a T-branched state");

    let mut full = Complex64::new(0.0, 0.0);
    for left in &branches {
        for right in &branches {
            let overlap = left.state.inner_product(&right.state).unwrap();
            full += left.weight.conj() * right.weight * overlap;
        }
    }

    let triangle = weighted_mps_norm_sq(&branches).unwrap();
    assert!(
        (triangle - full.re).abs() < 1e-12,
        "triangle {triangle} against full Gram {}",
        full.re
    );
}

fn phased(gate: Gate, angle: f64) -> Gate {
    let phase = Complex64::from_polar(1.0, angle);
    let m = gate.matrix_2x2();
    Gate::Fused(Box::new([
        [m[0][0] * phase, m[0][1] * phase],
        [m[1][0] * phase, m[1][1] * phase],
    ]))
}

fn assert_matches_statevector(circuit: &Circuit, tol: f64) -> StabRankResult {
    let sr = run_stabilizer_rank(circuit, 42).unwrap();
    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, circuit, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    for (i, (sr_p, sv_p)) in sr.probabilities.iter().zip(sv_probs.iter()).enumerate() {
        assert!(
            (sr_p - sv_p).abs() < tol,
            "prob[{i}]: stab_rank={sr_p}, statevector={sv_p}"
        );
    }
    sr
}

#[test]
fn fused_cliffords_with_a_global_phase_match_statevector() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(phased(Gate::H, 0.3), &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(phased(Gate::S, -0.7), &[1]);
    c.add_gate(phased(Gate::H, 1.9), &[2]);
    c.add_gate(phased(Gate::Tdg, 0.25), &[2]);
    c.add_gate(Gate::Cx, &[2, 1]);
    c.add_gate(phased(Gate::X, 2.0), &[1]);
    c.add_gate(phased(Gate::SXdg, -2.4), &[0]);

    let sr = assert_matches_statevector(&c, 1e-12);
    assert_eq!(sr.t_count, 2);
}

#[test]
fn rotations_on_the_pi_4_grid_expand_like_t() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rz(FRAC_PI_4), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::P(3.0 * FRAC_PI_4), &[1]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Rzz(-FRAC_PI_4), &[0, 1]);
    c.add_gate(Gate::Rz(FRAC_PI_2), &[0]);
    c.add_gate(Gate::Rx(FRAC_PI_4), &[1]);
    c.add_gate(Gate::H, &[0]);

    let sr = assert_matches_statevector(&c, 1e-12);
    assert_eq!(sr.t_count, 4);
    assert_eq!(c.t_count(), 4);
    assert!(c.is_clifford_plus_t());
}

#[test]
fn off_grid_rotations_are_rejected_by_gate_form() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rz(0.3), &[0]);
    let msg = format!("{:?}", run_stabilizer_rank(&c, 42).unwrap_err());
    assert!(msg.contains("rz") && msg.contains("pi/4"), "{msg}");
    assert!(!c.is_clifford_plus_t());

    let mut c = Circuit::new(1, 0);
    c.add_gate(
        Gate::Fused(Box::new(crate::circuit::openqasm::Parser::u_matrix(
            0.3, 0.7, -1.1,
        ))),
        &[0],
    );
    let msg = format!("{:?}", run_stabilizer_rank(&c, 42).unwrap_err());
    assert!(msg.contains("fused"), "{msg}");
}

#[test]
fn controlled_pauli_targets_lower_and_a_controlled_h_is_rejected() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::cu(Gate::X.matrix_2x2()), &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(
        Gate::cu(phased(Gate::Y, std::f64::consts::FRAC_PI_2).matrix_2x2()),
        &[1, 2],
    );
    c.add_gate(Gate::cu(Gate::Z.matrix_2x2()), &[2, 0]);
    c.add_gate(Gate::cu(Gate::S.matrix_2x2()), &[0, 2]);
    let sr = assert_matches_statevector(&c, 1e-12);
    assert_eq!(sr.t_count, 4);

    let mut ch = Circuit::new(2, 0);
    ch.add_gate(Gate::T, &[0]);
    ch.add_gate(Gate::cu(Gate::H.matrix_2x2()), &[0, 1]);
    let msg = format!("{:?}", run_stabilizer_rank(&ch, 42).unwrap_err());
    assert!(msg.contains("cu"), "{msg}");
}

#[test]
fn shots_lower_fused_and_guarded_gates() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(phased(Gate::H, 0.3), &[0]);
    c.add_gate(Gate::Rz(std::f64::consts::FRAC_PI_4), &[0]);
    c.add_gate(phased(Gate::H, -1.2), &[0]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(0),
        gate: phased(Gate::X, 2.0),
        targets: SmallVec::from_slice(&[1]),
    });
    c.add_measure(1, 1);

    let num_shots = 20_000;
    let result = run_stabilizer_rank_shots(&c, num_shots, 42).unwrap();
    assert!(result.shots.iter().all(|s| s[0] == s[1]));
    let p0 = result.shots.iter().filter(|s| !s[0]).count() as f64 / num_shots as f64;
    let expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
    assert!(
        (p0 - expected).abs() < 0.02,
        "P(0) = {p0}, expected {expected}"
    );
}
