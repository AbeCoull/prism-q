use super::*;
use crate::circuit::Circuit;
use crate::sim;

const EPS: f64 = 1e-12;

fn run_sparse(circuit: &Circuit) -> SparseBackend {
    let mut b = SparseBackend::new(42);
    sim::run_on(&mut b, circuit).unwrap();
    b
}

fn run_sparse_probs(circuit: &Circuit) -> Vec<f64> {
    let b = run_sparse(circuit);
    b.probabilities().unwrap()
}

#[test]
fn test_init_zero_state() {
    let mut b = SparseBackend::new(42);
    b.init(3, 0).unwrap();
    assert_eq!(b.state.len(), 1);
    assert!((b.state[&0].re - 1.0).abs() < EPS);
}

#[test]
fn test_x_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    assert!(b.state.contains_key(&1));
    assert!((b.state[&1].norm() - 1.0).abs() < EPS);
}

#[test]
fn test_h_creates_superposition() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 2);
    assert!((b.state[&0].norm_sqr() - 0.5).abs() < EPS);
    assert!((b.state[&1].norm_sqr() - 0.5).abs() < EPS);
}

#[test]
fn test_hh_is_identity() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[0]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    assert!((b.state[&0].re - 1.0).abs() < EPS);
}

#[test]
fn test_cx_bell_state() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 2);
    assert!((b.state[&0].norm_sqr() - 0.5).abs() < EPS);
    assert!((b.state[&3].norm_sqr() - 0.5).abs() < EPS);
}

#[test]
fn test_cz_phase() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    assert!((b.state[&3].re - (-1.0)).abs() < EPS);
}

#[test]
fn test_swap() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    assert!(b.state.contains_key(&1));
}

#[test]
fn test_rx_pi() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(std::f64::consts::PI), &[0]);
    let probs = run_sparse_probs(&c);
    assert!(probs[0].abs() < EPS);
    assert!((probs[1] - 1.0).abs() < EPS);
}

#[test]
fn test_rz_preserves_sparsity() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rz(1.234), &[0]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    assert!((b.state[&0].norm() - 1.0).abs() < EPS);
}

#[test]
fn test_measure_collapses() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    let outcome = b.classical_results()[0];
    if outcome {
        assert!(b.state.contains_key(&1));
    } else {
        assert!(b.state.contains_key(&0));
    }
}

#[test]
fn test_measure_deterministic() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);

    let b1 = run_sparse(&c);
    let b2 = run_sparse(&c);
    assert_eq!(b1.classical_results()[0], b2.classical_results()[0]);
}

#[test]
fn test_probs_bell() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let probs = run_sparse_probs(&c);
    assert!((probs[0] - 0.5).abs() < EPS);
    assert!(probs[1].abs() < EPS);
    assert!(probs[2].abs() < EPS);
    assert!((probs[3] - 0.5).abs() < EPS);
}

#[test]
fn test_probs_zero_state() {
    let c = Circuit::new(3, 0);
    let probs = run_sparse_probs(&c);
    assert!((probs[0] - 1.0).abs() < EPS);
    let rest: f64 = probs[1..].iter().sum();
    assert!(rest.abs() < EPS);
}

#[test]
fn test_pruning() {
    let mut b = SparseBackend::new(42);
    b.init(1, 0).unwrap();
    b.state.insert(1, Complex64::new(1e-20, 0.0));
    assert_eq!(b.state.len(), 2);
    b.prune();
    assert_eq!(b.state.len(), 1);
    assert!(b.state.contains_key(&0));
}

#[test]
fn test_fused_gate() {
    let h_mat = Gate::H.matrix_2x2();
    let t_mat = Gate::T.matrix_2x2();
    let zero = Complex64::new(0.0, 0.0);
    let mut fused = [[zero; 2]; 2];
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
    let p1 = run_sparse_probs(&c1);

    let mut c2 = Circuit::new(1, 0);
    c2.add_gate(Gate::Fused(Box::new(fused)), &[0]);
    let p2 = run_sparse_probs(&c2);

    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < EPS);
    }
}

#[test]
fn test_default_epsilon_reports_exact() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let b = run_sparse(&c);
    assert_eq!(b.epsilon, DEFAULT_EPSILON);
    assert_eq!(b.exactness(), crate::sim::Exactness::Exact);
}

#[test]
fn test_zero_epsilon_still_drops_exact_zeros() {
    let mut b = SparseBackend::new(42);
    b.set_prune_epsilon(0.0);
    b.init(1, 0).unwrap();
    b.state.insert(1, Complex64::new(0.0, 0.0));
    let zero = Complex64::new(0.0, 0.0);
    let half = Complex64::new(0.5, 0.0);
    b.apply_1q_matrix(0, &[[half, zero], [zero, half]]).unwrap();
    assert_eq!(b.state.len(), 1);
    assert!(b.state.contains_key(&0));
}

#[test]
fn test_lowering_epsilon_keeps_the_approximate_report() {
    let theta = 2.0 * (0.1_f64).sqrt().asin();
    let mut b = SparseBackend::new(42);
    b.set_prune_epsilon(0.2);
    b.init(1, 0).unwrap();
    b.apply_1q_matrix(0, &Gate::Ry(theta).matrix_2x2()).unwrap();
    b.set_prune_epsilon(DEFAULT_EPSILON);
    match b.exactness() {
        crate::sim::Exactness::Approximate {
            fidelity_lower_bound: Some(bound),
        } => assert!((bound - 0.9).abs() < EPS),
        other => panic!("lowering the threshold hid discarded weight: {other:?}"),
    }
}

#[test]
fn test_raised_epsilon_drops_weight_and_reports_bound() {
    // Ry puts weight 0.1 on |1>; a 0.2 threshold drops it at the prune
    // after the general-path gate.
    let theta = 2.0 * (0.1_f64).sqrt().asin();
    let mut b = SparseBackend::new(42);
    b.set_prune_epsilon(0.2);
    b.init(1, 0).unwrap();
    b.apply(&Instruction::Gate {
        gate: Gate::Ry(theta),
        targets: crate::circuit::SmallVec::from_slice(&[0]),
    })
    .unwrap();
    assert_eq!(b.state.len(), 1);
    assert!(b.state.contains_key(&0));
    match b.exactness() {
        crate::sim::Exactness::Approximate {
            fidelity_lower_bound: Some(bound),
        } => assert!((bound - 0.9).abs() < EPS),
        other => panic!("expected a bounded Approximate, got {other:?}"),
    }
}

// H puts 0.5 on each of |00>, |01>; Ry on q1 then splits each into
// 0.45 and 0.05. A 0.06 threshold drops both 0.05 entries, and the two
// kept entries must come back to 0.5 each rather than 0.45, which a
// unit-total CDF fold would otherwise hand to the highest kept state.
#[test]
fn test_prune_renormalizes_kept_entries() {
    let theta = 2.0 * (0.1_f64).sqrt().asin();
    let mut b = SparseBackend::new(42);
    b.set_prune_epsilon(0.06);
    b.init(2, 0).unwrap();
    b.apply_1q_matrix(0, &Gate::H.matrix_2x2()).unwrap();
    b.apply_1q_matrix(1, &Gate::Ry(theta).matrix_2x2()).unwrap();
    assert_eq!(b.state.len(), 2);

    let probs = b.probabilities().unwrap();
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < EPS, "probabilities sum to {total}");
    assert!((probs[0] - 0.5).abs() < EPS);
    assert!((probs[1] - 0.5).abs() < EPS);
    assert_eq!(probs[2], 0.0);
    assert_eq!(probs[3], 0.0);

    let z1 = b.pauli_expectations(&[vec![PauliTerm::z(1)]]).unwrap();
    assert!((z1[0] - 1.0).abs() < EPS);
    match b.exactness() {
        crate::sim::Exactness::Approximate {
            fidelity_lower_bound: Some(bound),
        } => assert!((bound - 0.9).abs() < EPS),
        other => panic!("expected a bounded Approximate, got {other:?}"),
    }
}

#[test]
fn test_epsilon_survives_init_weight_does_not() {
    let theta = 2.0 * (0.1_f64).sqrt().asin();
    let mut b = SparseBackend::new(42);
    b.set_prune_epsilon(0.2);
    b.init(1, 0).unwrap();
    b.apply_1q_matrix(0, &Gate::Ry(theta).matrix_2x2()).unwrap();
    assert!(b.pruned_weight > 0.0);
    b.init(1, 0).unwrap();
    match b.exactness() {
        crate::sim::Exactness::Approximate {
            fidelity_lower_bound: Some(bound),
        } => assert_eq!(bound, 1.0),
        other => panic!("expected Approximate after init, got {other:?}"),
    }
}

#[test]
#[should_panic(expected = "prune epsilon")]
fn test_prune_epsilon_rejects_one() {
    SparseBackend::new(42).set_prune_epsilon(1.0);
}

#[test]
fn test_y_gate_amplitude() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Y, &[0]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    assert!((b.state[&1] - Complex64::new(0.0, 1.0)).norm() < EPS);
}

// Distinct phases on the two antidiagonal coefficients, so a swapped
// a01/a10 assignment fails on amplitudes, not just norms.
#[test]
fn test_antidiagonal_fused_amplitudes() {
    let t = Gate::T.matrix_2x2();
    let zero = Complex64::new(0.0, 0.0);
    let fused = [[zero, t[1][1]], [t[0][0], zero]];

    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Fused(Box::new(fused)), &[0]);
    let b = run_sparse(&c);

    let h = 1.0 / 2.0_f64.sqrt();
    assert_eq!(b.state.len(), 2);
    assert!((b.state[&0] - t[1][1] * h).norm() < EPS);
    assert!((b.state[&1] - t[0][0] * h).norm() < EPS);
}

#[test]
fn test_antidiagonal_subunit_prunes() {
    let mut b = SparseBackend::new(42);
    b.init(1, 0).unwrap();
    b.state.insert(1, Complex64::new(1.5e-14, 0.0));
    let zero = Complex64::new(0.0, 0.0);
    let half = Complex64::new(0.5, 0.0);
    b.apply_1q_matrix(0, &[[zero, half], [half, zero]]).unwrap();
    assert_eq!(b.state.len(), 1);
    assert!(b.state.contains_key(&1));
}

#[test]
fn test_ghz_4_sparse() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 2);
    assert!((b.state[&0].norm_sqr() - 0.5).abs() < EPS);
    assert!((b.state[&15].norm_sqr() - 0.5).abs() < EPS);
}

#[test]
fn test_cu_phase_applies_phase() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_4), &[0, 1]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 1);
    let expected = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    assert!((b.state[&3] - expected).norm() < EPS);
}

#[test]
fn test_cu_phase_no_action_control_zero() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::cphase(1.0), &[0, 1]);
    let b = run_sparse(&c);
    let h = 1.0 / 2.0_f64.sqrt();
    assert!((b.state[&0].re - h).abs() < EPS);
    assert!((b.state[&2].re - h).abs() < EPS);
    assert!(!b.state.contains_key(&1));
    assert!(!b.state.contains_key(&3));
}

#[test]
fn test_cu_phase_matches_cz() {
    let mut c1 = Circuit::new(2, 0);
    c1.add_gate(Gate::H, &[0]);
    c1.add_gate(Gate::H, &[1]);
    c1.add_gate(Gate::cphase(std::f64::consts::PI), &[0, 1]);

    let mut c2 = Circuit::new(2, 0);
    c2.add_gate(Gate::H, &[0]);
    c2.add_gate(Gate::H, &[1]);
    c2.add_gate(Gate::Cz, &[0, 1]);

    let b1 = run_sparse(&c1);
    let b2 = run_sparse(&c2);

    for (&idx, &amp1) in &b1.state {
        let amp2 = b2
            .state
            .get(&idx)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0));
        assert!((amp1 - amp2).norm() < EPS, "mismatch at idx {idx}");
    }
}

#[test]
fn test_diagonal_1q_in_place() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::P(1.234), &[0]);
    let b = run_sparse(&c);
    assert_eq!(b.state.len(), 2);
    let h = 1.0 / 2.0_f64.sqrt();
    assert!((b.state[&0].re - h).abs() < EPS);
    assert!((b.state[&1] - Complex64::from_polar(h, 1.234)).norm() < EPS);
}

#[test]
fn test_batch_rzz_matches_individual() {
    use crate::gates::BatchRzzData;

    let edges = vec![(0usize, 1usize, 0.7f64), (1, 2, 1.3)];

    let mut c1 = Circuit::new(3, 0);
    for q in 0..3 {
        c1.add_gate(Gate::H, &[q]);
    }
    for &(q0, q1, theta) in &edges {
        c1.add_gate(Gate::Rzz(theta), &[q0, q1]);
    }
    let b1 = run_sparse(&c1);

    let mut c2 = Circuit::new(3, 0);
    for q in 0..3 {
        c2.add_gate(Gate::H, &[q]);
    }
    c2.add_gate(Gate::BatchRzz(Box::new(BatchRzzData { edges })), &[0, 1, 2]);
    let b2 = run_sparse(&c2);

    for (&idx, &amp1) in &b1.state {
        let amp2 = b2
            .state
            .get(&idx)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0));
        assert!((amp1 - amp2).norm() < EPS, "mismatch at idx {idx}");
    }
}

#[test]
fn test_diagonal_batch_matches_individual() {
    use crate::gates::{DiagEntry, DiagonalBatchData};

    let mut c1 = Circuit::new(3, 0);
    for q in 0..3 {
        c1.add_gate(Gate::H, &[q]);
    }
    c1.add_gate(Gate::S, &[0]);
    c1.add_gate(Gate::Cz, &[0, 1]);
    c1.add_gate(Gate::Rzz(0.9), &[1, 2]);
    let b1 = run_sparse(&c1);

    let s_mat = Gate::S.matrix_2x2();
    let mut c2 = Circuit::new(3, 0);
    for q in 0..3 {
        c2.add_gate(Gate::H, &[q]);
    }
    c2.add_gate(
        Gate::DiagonalBatch(Box::new(DiagonalBatchData {
            entries: vec![
                DiagEntry::Phase1q {
                    qubit: 0,
                    d0: s_mat[0][0],
                    d1: s_mat[1][1],
                },
                DiagEntry::Phase2q {
                    q0: 0,
                    q1: 1,
                    phase: Complex64::new(-1.0, 0.0),
                },
                DiagEntry::Parity2q {
                    q0: 1,
                    q1: 2,
                    same: Complex64::from_polar(1.0, -0.45),
                    diff: Complex64::from_polar(1.0, 0.45),
                },
            ],
        })),
        &[0, 1, 2],
    );
    let b2 = run_sparse(&c2);

    for (&idx, &amp1) in &b1.state {
        let amp2 = b2
            .state
            .get(&idx)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0));
        assert!((amp1 - amp2).norm() < EPS, "mismatch at idx {idx}");
    }
}

#[test]
fn test_batch_phase_matches_individual() {
    use crate::gates::BatchPhaseData;
    use smallvec::smallvec;

    let phase1 = Complex64::from_polar(1.0, 0.5);
    let phase2 = Complex64::from_polar(1.0, 1.2);

    let mut c1 = Circuit::new(3, 0);
    c1.add_gate(Gate::H, &[0]);
    c1.add_gate(Gate::H, &[1]);
    c1.add_gate(Gate::H, &[2]);
    c1.add_gate(Gate::cphase(0.5), &[0, 1]);
    c1.add_gate(Gate::cphase(1.2), &[0, 2]);
    let p1 = run_sparse_probs(&c1);

    let mut c2 = Circuit::new(3, 0);
    c2.add_gate(Gate::H, &[0]);
    c2.add_gate(Gate::H, &[1]);
    c2.add_gate(Gate::H, &[2]);
    c2.add_gate(
        Gate::BatchPhase(Box::new(BatchPhaseData {
            phases: smallvec![(1, phase1), (2, phase2)],
        })),
        &[0, 1, 2],
    );
    let p2 = run_sparse_probs(&c2);

    for (a, b) in p1.iter().zip(p2.iter()) {
        assert!((a - b).abs() < EPS, "probs mismatch: {a} vs {b}");
    }
}
