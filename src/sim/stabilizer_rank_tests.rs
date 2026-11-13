use super::*;
use crate::circuit::{ClassicalCondition, SmallVec};
use crate::limits::ResolvedLimits;

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

    let public_result = crate::sim::run_shots_with(
        crate::sim::BackendKind::StabilizerRank,
        &c,
        8,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
    assert_eq!(public_result.shots.len(), 8);

    let auto_result = crate::sim::run_shots_with(
        crate::sim::BackendKind::Auto,
        &c,
        8,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
    assert_eq!(auto_result.shots.len(), 8);
}

#[test]
fn shots_decline_a_register_past_its_cap() {
    let width = MAX_REGISTER_QUBITS + 1;
    let mut c = Circuit::new(width + 1, width + 1);
    for q in 0..width {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    for q in 0..width {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..=width {
        c.add_measure(q, q);
    }

    assert!(terminal_register_price(&c, MAX_REGISTER_QUBITS).is_none());
    let err = run_stabilizer_rank_shots(&c, 8, 42).unwrap_err();
    assert!(
        matches!(err, PrismError::BackendUnsupported { .. }),
        "got {err:?}"
    );
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
    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &c,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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
        let sv = crate::sim::run_with(
            crate::sim::BackendKind::Statevector,
            &c,
            42,
            &ResolvedLimits::process_default(),
        )
        .unwrap();
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
    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &c,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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
    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &c,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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
    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &c,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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
fn twenty_one_t_on_one_qubit_hold_two_terms() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    for _ in 0..21 {
        c.add_gate(Gate::T, &[0]);
    }
    c.add_gate(Gate::H, &[0]);
    let sr = assert_matches_statevector(&c, 1e-10);
    assert_eq!(sr.t_count, 21);
    assert_eq!(sr.num_terms, 2);
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

fn classical_fidelity(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q)
        .map(|(a, b)| (a * b).sqrt())
        .sum::<f64>()
        .powi(2)
}

// Asserts only what holds for every order of tied coefficients: the cut norms and the
// distribution move with the tie split once cuts follow one another.
fn assert_inside_the_bound(exact: &StabRankResult, approx: &StabRankResult, budget: usize) {
    let label = format!("budget {budget}");
    assert!(approx.num_terms <= budget, "{label}");
    let total: f64 = approx.probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "{label}: summed to {total}");
    let floor = approx.fidelity_bound();
    let deviation: f64 = exact
        .probabilities
        .iter()
        .zip(&approx.probabilities)
        .map(|(e, a)| (e - a).abs())
        .sum();
    let bound = 2.0 * (1.0 - floor).sqrt();
    assert!(
        deviation <= bound + 1e-9,
        "{label}: deviated {deviation} past {bound}"
    );
    let bc2 = classical_fidelity(&exact.probabilities, &approx.probabilities);
    assert!(
        bc2 >= floor - 1e-9,
        "{label}: {bc2} below the floor {floor}"
    );
    assert!(approx.kept_norm_sqr <= 1.0 + 1e-12, "{label}");
    assert!(
        1.0 - approx.kept_norm_sqr <= approx.discarded_weight.powi(2) + 1e-12,
        "{label}"
    );
    assert!(
        floor >= 1.0 - approx.discarded_weight.powi(2) - 1e-12,
        "{label}"
    );
    if approx.pruned_count == 0 {
        assert_eq!(approx.discarded_weight, 0.0, "{label}");
        assert_eq!(approx.kept_norm_sqr, 1.0, "{label}");
        assert_eq!(floor, 1.0, "{label}");
        for (e, a) in exact.probabilities.iter().zip(&approx.probabilities) {
            assert!((e - a).abs() < 1e-10, "{label}");
        }
    }
}

#[test]
fn pruned_runs_stay_inside_the_reported_error_bound() {
    let c = eight_t_circuit();
    let exact = run_stabilizer_rank(&c, 42).unwrap();
    assert_eq!(exact.num_terms, 16);

    for (budget, floor) in [(4usize, 0.6), (8, 0.94), (12, 0.99)] {
        let approx = run_stabilizer_rank_approx(&c, budget, 42).unwrap();
        assert!(approx.pruned_count > 0);
        assert_inside_the_bound(&exact, &approx, budget);
        assert!(approx.fidelity_bound() > floor, "budget {budget}");
    }
}

#[test]
fn budgets_past_the_register_are_exact() {
    let c = eight_t_circuit();
    let exact = run_stabilizer_rank(&c, 42).unwrap();
    for budget in [16usize, 64, 160, 224] {
        let approx = run_stabilizer_rank_approx(&c, budget, 42).unwrap();
        assert_eq!(approx.pruned_count, 0);
        assert_eq!(approx.discarded_weight, 0.0);
        assert_eq!(approx.fidelity_bound(), 1.0);
        assert_inside_the_bound(&exact, &approx, budget);
    }
}

// The cut falls between the c^2 s and c s^2 coefficient groups, so it is tie invariant.
#[test]
fn one_cut_reports_the_exact_fidelity() {
    let mut c = Circuit::new(3, 0);
    for q in 0..3 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
        c.add_gate(Gate::H, &[q]);
    }
    let exact = run_stabilizer_rank(&c, 42).unwrap();
    let approx = run_stabilizer_rank_approx(&c, 4, 42).unwrap();
    let (s, co) = std::f64::consts::FRAC_PI_8.sin_cos();
    let want = co.powi(6) + 3.0 * co.powi(4) * s * s;
    assert_eq!(approx.pruned_count, 4);
    assert_eq!(approx.num_terms, 4);
    assert!((approx.kept_norm_sqr - want).abs() < 1e-12);
    assert!((approx.fidelity_bound() - want).abs() < 1e-12);
    assert!((1.0 - approx.discarded_weight.powi(2) - want).abs() < 1e-12);
    let bc2 = classical_fidelity(&exact.probabilities, &approx.probabilities);
    assert!((bc2 - want).abs() < 1e-12, "{bc2} vs {want}");
    assert_inside_the_bound(&exact, &approx, 4);
}

#[test]
fn seeded_pruned_runs_stay_inside_the_bound() {
    for seed in 42..47u64 {
        let c = crate::circuits::clifford_t_circuit(8, 6, 0.25, seed);
        let exact = run_stabilizer_rank(&c, 42).unwrap();
        for budget in [8usize, 32, 128] {
            let approx = run_stabilizer_rank_approx(&c, budget, 42).unwrap();
            assert_inside_the_bound(&exact, &approx, budget);
        }
    }
}

// The `stabilizer_rank/approx_*` bench circuit.
fn t_insertion_circuit(n: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        if rng.random_bool(0.5) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    for _ in 0..t_count {
        let q = rng.random_range(0..n);
        c.add_gate(Gate::T, &[q]);
    }
    for _ in 0..3 {
        for q in 0..n {
            let g = rng.random_range(0..cliffords.len());
            c.add_gate(cliffords[g].clone(), &[q]);
        }
        for q in 0..n - 1 {
            if rng.random_bool(0.5) {
                c.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    c
}

#[test]
fn bench_circuits_certify_a_floor_at_budget_128() {
    for (t, floor) in [(16usize, 0.8), (20, 0.5)] {
        let c = t_insertion_circuit(10, t, 0xDEAD_BEEF);
        let exact = run_stabilizer_rank(&c, 42).unwrap();
        let approx = run_stabilizer_rank_approx(&c, 128, 42).unwrap();
        assert!(approx.pruned_count > 0);
        assert_inside_the_bound(&exact, &approx, 128);
        assert!(
            approx.fidelity_bound() > floor,
            "{t} T: floor {}",
            approx.fidelity_bound()
        );
    }

    let c = t_insertion_circuit(10, 16, 0xDEAD_BEEF);
    let one_cut = run_stabilizer_rank_approx(&c, 256, 42).unwrap();
    assert!((one_cut.fidelity_bound() - one_cut.kept_norm_sqr).abs() < 1e-12);
    assert!((one_cut.kept_norm_sqr - 0.988898).abs() < 1e-6);
}

#[test]
fn pruned_route_refuses_a_register_past_its_width() {
    let width = MAX_REGISTER_QUBITS + 1;
    let mut c = Circuit::new(width, 0);
    for q in 0..width {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    let err = run_stabilizer_rank_approx(&c, 128, 42).unwrap_err();
    assert!(
        matches!(err, PrismError::BackendUnsupported { .. }),
        "got {err:?}"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn pruned_route_is_identical_on_one_and_many_threads() {
    let c = t_insertion_circuit(10, 20, 0xDEAD_BEEF);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run_stabilizer_rank_approx(&c, 128, 42).unwrap())
    };
    let (one, many) = (run(1), run(4));
    assert_eq!(one.probabilities, many.probabilities);
    assert_eq!(one.discarded_weight, many.discarded_weight);
    assert_eq!(one.kept_norm_sqr, many.kept_norm_sqr);
    assert_eq!(
        (one.num_terms, one.pruned_count),
        (many.num_terms, many.pruned_count)
    );
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

    let mut branches = initial_mps_branches(&c, 42).unwrap();
    for inst in &c.instructions {
        let Instruction::Gate { gate, targets } = inst else {
            unreachable!("the circuit holds gates only");
        };
        apply_mps_gate(&mut branches, gate, targets).unwrap();
    }
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
    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        circuit,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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

#[test]
fn rejects_too_many_qubits() {
    let n = MAX_STATEVECTOR_QUBITS + 1;
    let c = Circuit::new(n, 0);
    assert!(run_stabilizer_rank(&c, 0).is_err());
    assert!(run_stabilizer_rank_approx(&c, 0, 16).is_err());
}

#[test]
fn approx_path_runs_with_pruning() {
    let mut c = Circuit::new(3, 0);
    for q in 0..3 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    let result = run_stabilizer_rank_approx(&c, 4, 4).unwrap();
    assert!(result.pruned_count > 0);
    let total: f64 = result.probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-9);
}

#[test]
fn approx_path_tdg_runs() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Tdg, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Tdg, &[1]);
    let result = run_stabilizer_rank_approx(&c, 7, 16).unwrap();
    assert!(result.probabilities.iter().all(|p| p.is_finite()));
}

// A Clifford circuit on `n` qubits drawn from `rng`: `2 * n` layers of one
// single-qubit gate and one entangler each, enough for the two states of a
// pair to share no structure.
fn random_clifford(n: usize, rng: &mut ChaCha8Rng) -> Circuit {
    let singles = [
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::SX,
    ];
    let pairs = [Gate::Cx, Gate::Cz, Gate::Swap];
    let mut circuit = Circuit::new(n, 0);
    for _ in 0..2 * n {
        circuit.add_gate(
            singles[rng.random_range(0..singles.len())].clone(),
            &[rng.random_range(0..n)],
        );
        let a = rng.random_range(0..n);
        let b = (a + 1 + rng.random_range(0..n - 1)) % n;
        circuit.add_gate(pairs[rng.random_range(0..pairs.len())].clone(), &[a, b]);
    }
    circuit
}

fn tableau_of(circuit: &Circuit) -> StabilizerBackend {
    let mut backend = StabilizerBackend::new(42);
    crate::sim::run_on(&mut backend, circuit).unwrap();
    backend
}

fn dense_overlap_sq(a: &StabilizerBackend, b: &StabilizerBackend) -> f64 {
    let (x, y) = (
        a.export_statevector().unwrap(),
        b.export_statevector().unwrap(),
    );
    let inner: Complex64 = x.iter().zip(&y).map(|(p, q)| p.conj() * q).sum();
    inner.norm_sqr()
}

// Two unrelated Clifford states, not one state against a perturbation of
// itself: rows drawn from two different stabilizer groups can anticommute, and
// their product then carries a factor of i that a merged row has to keep. The
// pair below is the smallest one where the elimination reaches such a product,
// and rounding the factor to a sign reads 1/2 on two orthogonal states.
#[test]
fn overlap_of_two_orthogonal_clifford_states_is_zero() {
    let mut left = Circuit::new(2, 0);
    left.add_gate(Gate::H, &[0]);
    left.add_gate(Gate::H, &[1]);
    left.add_gate(Gate::Cz, &[0, 1]);
    left.add_gate(Gate::Sdg, &[0]);
    left.add_gate(Gate::Cz, &[0, 1]);

    let mut right = Circuit::new(2, 0);
    right.add_gate(Gate::H, &[0]);
    right.add_gate(Gate::Cz, &[0, 1]);
    right.add_gate(Gate::S, &[0]);
    right.add_gate(Gate::Cx, &[0, 1]);

    let (a, b) = (tableau_of(&left), tableau_of(&right));
    assert!(
        dense_overlap_sq(&a, &b) < 1e-12,
        "the fixture is meant to be an orthogonal pair"
    );
    let overlap = stabilizer_overlap_sq(&a, &b, 2);
    assert!(
        overlap < 1e-12,
        "overlap of an orthogonal pair reads {overlap}"
    );
}

// 400 unrelated pairs across widths 2 to 5, each against the dense inner
// product of the same two tableaux. A merged row whose phase is rounded to a
// sign fails about 3% of these.
#[test]
fn overlap_matches_the_dense_inner_product_over_random_clifford_pairs() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for n in 2..=5 {
        for case in 0..100 {
            let left = random_clifford(n, &mut rng);
            let right = random_clifford(n, &mut rng);
            let (a, b) = (tableau_of(&left), tableau_of(&right));
            let want = dense_overlap_sq(&a, &b);
            let got = stabilizer_overlap_sq(&a, &b, n);
            assert!(
                (got - want).abs() < 1e-12,
                "{n} qubits, pair {case}: tableau overlap {got} against dense {want}"
            );
        }
    }
}

/// `H` on every qubit, then `t` rotations (`T` or `Tdg`) spread through random
/// Cliffords.
fn scrambled_clifford_t(n: usize, t: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::SX, Gate::X, Gate::Z];
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for _ in 0..t {
        for _ in 0..n {
            if rng.random_bool(0.4) {
                let a = rng.random_range(0..n);
                let b = (a + rng.random_range(1..n)) % n;
                let gate = [Gate::Cx, Gate::Cz, Gate::Swap][rng.random_range(0..3)].clone();
                c.add_gate(gate, &[a, b]);
            } else {
                let gate = cliffords[rng.random_range(0..cliffords.len())].clone();
                c.add_gate(gate, &[rng.random_range(0..n)]);
            }
        }
        let gate = if rng.random_bool(0.5) {
            Gate::T
        } else {
            Gate::Tdg
        };
        c.add_gate(gate, &[rng.random_range(0..n)]);
    }
    c
}

/// The shape of the `stabilizer_rank` bench rows: an `H` layer and a CX ladder, every
/// `T`, then three Clifford layers.
fn layered_clifford_t(n: usize, t: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        if rng.random_bool(0.5) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    for _ in 0..t {
        c.add_gate(Gate::T, &[rng.random_range(0..n)]);
    }
    for _ in 0..3 {
        for q in 0..n {
            c.add_gate(
                cliffords[rng.random_range(0..cliffords.len())].clone(),
                &[q],
            );
        }
        for q in 0..n - 1 {
            if rng.random_bool(0.5) {
                c.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    c
}

fn assert_merged_terms(sr: &StabRankResult, n: usize, t: usize) {
    assert_eq!(sr.t_count, t);
    assert!(sr.num_terms.is_power_of_two());
    assert!(sr.num_terms <= 1 << n.min(t), "{} terms", sr.num_terms);
}

#[test]
fn scrambled_circuits_past_twenty_t_match_statevector() {
    for (n, t) in [(10, 24), (12, 30), (14, 30)] {
        let c = scrambled_clifford_t(n, t, 42);
        let sr = assert_matches_statevector(&c, 1e-10);
        assert_merged_terms(&sr, n, t);
    }
}

#[test]
fn layered_circuits_at_thirty_t_match_statevector() {
    for n in [12, 16] {
        let c = layered_clifford_t(n, 30, 42);
        let sr = assert_matches_statevector(&c, 1e-10);
        assert_merged_terms(&sr, n, 30);
    }
}

#[test]
fn more_t_than_qubits_keeps_the_register_at_the_qubit_count() {
    let c = scrambled_clifford_t(4, 12, 42);
    let sr = assert_matches_statevector(&c, 1e-10);
    assert_merged_terms(&sr, 4, 12);
    assert!(sr.num_terms <= 16);
}

#[test]
fn declined_register_falls_back_to_one_branch_per_t() {
    let c = eight_t_circuit();
    let merged = run_stabilizer_rank(&c, 42).unwrap();
    let expanded = exact_probabilities(&c, 42, 2).unwrap();
    assert_eq!(merged.num_terms, 16);
    assert_eq!(expanded.num_terms, 256);
    assert_eq!(expanded.t_count, 8);
    for (i, (m, e)) in merged
        .probabilities
        .iter()
        .zip(&expanded.probabilities)
        .enumerate()
    {
        assert!((m - e).abs() < 1e-10, "prob[{i}]: merged {m}, expanded {e}");
    }
}

// Mid-circuit shots on the deferred register, against the branch sampler they replace and
// against a dense run.

fn conditional(c: &mut Circuit, condition: ClassicalCondition, gate: Gate, qubit: usize) {
    c.instructions.push(Instruction::Conditional {
        condition,
        gate,
        targets: SmallVec::from_slice(&[qubit]),
    });
}

/// `H` on every qubit, three layers of random one-qubit Cliffords and random neighbour CX with
/// `t` T gates spread over them, then qubit 0 measured into bit 0, `guard` on qubit 1 when
/// that bit is one, qubit 0 reset and coupled back to qubit 1, and qubits 0 and 1 measured
/// into bits 2 and 1.
fn mid_circuit_case(n: usize, t: usize, guard: Gate, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let ones = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Z];
    let mut c = Circuit::new(n, 3);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for layer in 0..3 {
        for i in (layer..t).step_by(3) {
            c.add_gate(Gate::T, &[(i * 5 + layer) % n]);
        }
        for q in 0..n {
            c.add_gate(ones[rng.random_range(0..ones.len())].clone(), &[q]);
        }
        for _ in 0..n / 2 {
            let a = rng.random_range(0..n - 1);
            c.add_gate(Gate::Cx, &[a, a + 1]);
        }
    }
    c.add_measure(0, 0);
    conditional(&mut c, ClassicalCondition::BitIsOne(0), guard, 1);
    c.add_reset(0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 2);
    c.add_measure(1, 1);
    c
}

/// Qubit 0 measured and reset twice, with gates on the fresh qubit between.
fn twice_reset_case() -> Circuit {
    let mut c = Circuit::new(3, 4);
    for q in 0..3 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::H, &[1]);
    c.add_measure(0, 0);
    c.add_reset(0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 2]);
    c.add_measure(0, 1);
    c.add_reset(0);
    c.add_gate(Gate::Cx, &[2, 0]);
    c.add_gate(Gate::H, &[2]);
    c.add_measure(0, 2);
    c.add_measure(2, 3);
    c
}

/// An `X` on qubit 2 guarded by the parity of two measured bits.
fn parity_guard_case() -> Circuit {
    let mut c = Circuit::new(4, 3);
    for q in 0..4 {
        c.add_gate(Gate::H, &[q]);
    }
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Tdg, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::H, &[3]);
    c.add_measure(0, 0);
    c.add_measure(3, 1);
    conditional(
        &mut c,
        ClassicalCondition::Parity {
            bits: vec![0, 1].into(),
            expected: true,
        },
        Gate::X,
        2,
    );
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::H, &[2]);
    c.add_measure(2, 2);
    c
}

fn deferral_cases() -> Vec<(String, Circuit, usize)> {
    // The branch sampler pays `B (B + 1)` MPS overlaps per measurement at `B = 2^t`
    // branches, so the shot count falls as `t` and the width grow.
    let mut cases = Vec::new();
    for (n, t, shots) in [(8, 2, 2000), (8, 4, 1000), (12, 2, 1000), (12, 4, 1000)] {
        let circuit = mid_circuit_case(n, t, Gate::X, 42);
        cases.push((format!("{n}q {t}t"), circuit, shots));
    }
    cases.push(("8q 6t".into(), mid_circuit_case(8, 6, Gate::X, 42), 400));
    cases.push(("guarded z".into(), mid_circuit_case(6, 3, Gate::Z, 7), 2000));
    cases.push((
        "guarded y".into(),
        mid_circuit_case(6, 3, Gate::Y, 11),
        2000,
    ));
    cases.push(("reset twice".into(), twice_reset_case(), 2000));
    cases.push(("parity guard".into(), parity_guard_case(), 2000));
    cases
}

/// Per-outcome frequencies of two samples of `shots` agree within five standard deviations
/// of their difference plus one shot, `p` their pooled frequency.
fn assert_same_distribution(
    label: &str,
    new: &crate::sim::ShotsResult,
    old: &crate::sim::ShotsResult,
) {
    let shots = new.shots.len();
    assert_eq!(old.shots.len(), shots, "{label}: shot totals");
    let (new, old) = (new.counts(), old.counts());
    let keys: std::collections::BTreeSet<&Vec<u64>> = new.keys().chain(old.keys()).collect();
    for key in keys {
        let a = new.get(key).copied().unwrap_or(0) as f64 / shots as f64;
        let b = old.get(key).copied().unwrap_or(0) as f64 / shots as f64;
        let p = (a + b) / 2.0;
        let bound = 5.0 * (p * (1.0 - p) * 2.0 / shots as f64).sqrt() + 1.0 / shots as f64;
        assert!(
            (a - b).abs() <= bound,
            "{label}: outcome {key:?} at {a} against {b}"
        );
    }
}

#[test]
#[cfg_attr(
    prism_lowered_thresholds,
    ignore = "the lowered register cap is narrower than these registers"
)]
fn deferred_register_shots_match_the_branch_sampler() {
    for (label, circuit, shots) in deferral_cases() {
        let lowered = lower_clifford_t_circuit(&circuit).unwrap();
        assert!(
            deferred_sampler(&lowered).unwrap().is_some(),
            "{label}: the deferral declined"
        );
        let new = run_stabilizer_rank_shots(&circuit, shots, 42).unwrap();
        let old = sample_mps_branches_online(&circuit, shots, 42).unwrap();
        assert_same_distribution(&label, &new, &old);
    }
}

/// Exact record distribution of a dynamic circuit, indexed by classical bits, from a dense
/// run of its dilation: a CX copies each measured qubit onto a fresh ancilla, a reset hands
/// the qubit's later gates to a fresh wire, and a guarded Pauli is applied controlled by
/// the ancillas its guard reads.
fn dilated_distribution(circuit: &Circuit) -> Vec<f64> {
    let events = circuit
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Measure { .. } | Instruction::Reset { .. }))
        .count();
    let mut unitary = Circuit::new(circuit.num_qubits + events, 0);
    let mut wire: Vec<usize> = (0..circuit.num_qubits).collect();
    let mut bit_wire: Vec<Option<usize>> = vec![None; circuit.num_classical_bits];
    let mut next = circuit.num_qubits;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                let mapped: Vec<usize> = targets.iter().map(|&q| wire[q]).collect();
                unitary.add_gate(gate.clone(), &mapped);
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                unitary.add_gate(Gate::Cx, &[wire[*qubit], next]);
                bit_wire[*classical_bit] = Some(next);
                next += 1;
            }
            Instruction::Reset { qubit } => {
                wire[*qubit] = next;
                next += 1;
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                let bits: Vec<usize> = match condition {
                    ClassicalCondition::BitIsOne(bit) => vec![*bit],
                    ClassicalCondition::Parity {
                        bits,
                        expected: true,
                    } => bits.to_vec(),
                    other => panic!("no dilation for {other:?}"),
                };
                let controlled = Gate::cu(gate.matrix_2x2());
                for bit in bits {
                    if let Some(control) = bit_wire[bit] {
                        unitary.add_gate(controlled.clone(), &[control, wire[targets[0]]]);
                    }
                }
            }
            other => panic!("no dilation for {other:?}"),
        }
    }
    let probs = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &unitary,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap()
    .probabilities
    .unwrap()
    .to_vec();
    let mut dist = vec![0.0; 1 << circuit.num_classical_bits];
    for (index, p) in probs.into_iter().enumerate() {
        let key = bit_wire
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_some_and(|w| (index >> w) & 1 == 1))
            .fold(0usize, |acc, (bit, _)| acc | 1 << bit);
        dist[key] += p;
    }
    dist
}

#[test]
#[cfg_attr(
    prism_lowered_thresholds,
    ignore = "the lowered register cap is narrower than these registers"
)]
fn deferred_register_probabilities_match_a_dense_dilation() {
    let cases = [
        ("guarded x", mid_circuit_case(5, 4, Gate::X, 3)),
        ("guarded z", mid_circuit_case(6, 3, Gate::Z, 7)),
        ("guarded y", mid_circuit_case(6, 3, Gate::Y, 11)),
        ("reset twice", twice_reset_case()),
        ("parity guard", parity_guard_case()),
    ];
    for (label, circuit) in cases {
        let deferred = crate::sim::compiled::defer_clifford_t_circuit(&circuit).unwrap();
        let sampler = TerminalSampler::compile(&deferred, MAX_REGISTER_QUBITS)
            .unwrap()
            .unwrap();
        let want = dilated_distribution(&circuit);
        for (key, (got, want)) in sampler.distribution().iter().zip(&want).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "{label}: outcome {key}: got {got}, want {want}"
            );
        }
    }
}

/// `mid_circuit_case` with `gate` guarded in place of the Pauli, or with a gate on the
/// measured qubit before its reset, or with a measurement inside a guarded body.
fn declined_cases(n: usize, t: usize) -> Vec<(&'static str, Circuit)> {
    let mut guarded_t = mid_circuit_case(n, t, Gate::X, 5);
    let at = guarded_t
        .instructions
        .iter()
        .position(|i| matches!(i, Instruction::Conditional { .. }))
        .unwrap();
    guarded_t.instructions[at] = Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::T,
        targets: SmallVec::from_slice(&[1]),
    };
    let mut guarded_s = guarded_t.clone();
    guarded_s.instructions[at] = Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::S,
        targets: SmallVec::from_slice(&[1]),
    };
    let mut reused = mid_circuit_case(n, t, Gate::X, 5);
    reused.instructions.insert(
        at,
        Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[0]),
        },
    );
    let mut region = mid_circuit_case(n, t, Gate::X, 5);
    region.instructions[at] = Instruction::Region(Box::new(GuardedRegion::new(
        ClassicalCondition::BitIsOne(0),
        vec![Instruction::Measure {
            qubit: 1,
            classical_bit: 1,
        }],
    )));
    vec![
        ("guarded t", guarded_t),
        ("guarded s", guarded_s),
        ("measured qubit reused", reused),
        ("measurement in a guarded body", region),
    ]
}

#[test]
fn deferral_declines_what_it_cannot_move() {
    for (label, circuit) in declined_cases(6, 2) {
        assert!(
            deferred_register_circuit(&circuit).is_none(),
            "{label}: the deferral accepted"
        );
        let new = run_stabilizer_rank_shots(&circuit, 400, 42).unwrap();
        let old = sample_mps_branches_online(&circuit, 400, 42).unwrap();
        assert_eq!(
            new.shots, old.shots,
            "{label}: the branch sampler is unchanged"
        );
    }
}

#[test]
fn branch_overflow_fails_before_the_first_shot() {
    let mut circuit = Circuit::new(24, 2);
    for q in 0..21 {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit.add_measure(0, 0);
    conditional(&mut circuit, ClassicalCondition::BitIsOne(0), Gate::S, 1);
    circuit.add_measure(1, 1);
    let err = run_stabilizer_rank_shots(&circuit, 1, 42).unwrap_err();
    match err {
        PrismError::BackendUnsupported { operation, .. } => {
            assert!(operation.contains("21 T gates"), "{operation}");
        }
        other => panic!("expected BackendUnsupported, got {other:?}"),
    }
}
