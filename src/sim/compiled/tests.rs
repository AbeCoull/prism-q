use super::*;
use crate::circuits;
use crate::gates::Gate;
use crate::sim::BackendKind;

#[test]
fn ghz_rank_is_one() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 1, "GHZ-10 should have rank 1");
}

#[test]
fn bell_pairs_rank() {
    let n_pairs = 5;
    let mut c = circuits::independent_bell_pairs(n_pairs);
    let n = c.num_qubits;
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), n_pairs, "5 Bell pairs should have rank 5");
}

#[test]
fn random_clifford_rank_is_n() {
    let n = 10;
    let mut c = circuits::clifford_heavy_circuit(n, 50, 42);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(
        sampler.rank(),
        n,
        "Random Clifford 10q d50 should have rank {n}"
    );
}

#[test]
fn non_clifford_rejected() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::Rx(0.5), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    let result = compile_measurements(&c, 42);
    assert!(result.is_err());
}

#[test]
fn no_measurements_rank_zero() {
    let c = circuits::ghz_circuit(5);
    let sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 0);
    assert_eq!(sampler.num_measurements(), 0);
}

#[test]
fn identity_circuit_all_zeros() {
    let mut c = Circuit::new(4, 4);
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 0, "Identity circuit should have rank 0");

    let mut sampler = sampler;
    for _ in 0..100 {
        let outcome = sampler.sample();
        assert!(outcome.iter().all(|&b| !b), "All outcomes should be 0");
    }
}

#[test]
fn single_h_measure() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    let sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 1, "H+measure should have rank 1");
}

#[test]
fn ghz_distribution() {
    let n = 10;
    let mut c = circuits::ghz_circuit(n);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let result = run_shots_compiled(&c, 10_000, 42).unwrap();
    let counts = result.counts();

    let all_zero: Vec<bool> = vec![false; n];
    let all_one: Vec<bool> = vec![true; n];

    let n_zero = counts.get(&all_zero).copied().unwrap_or(0);
    let n_one = counts.get(&all_one).copied().unwrap_or(0);

    assert_eq!(
        counts.len(),
        2,
        "GHZ should produce exactly 2 outcomes, got {}",
        counts.len()
    );
    assert!(
        n_zero + n_one == 10_000,
        "All shots should be all-0 or all-1"
    );
    let ratio = n_zero as f64 / 10_000.0;
    assert!(
        (0.45..=0.55).contains(&ratio),
        "Expected ~50/50, got {ratio:.3}"
    );
}

#[test]
fn bell_pairs_always_agree() {
    let n_pairs = 5;
    let mut c = circuits::independent_bell_pairs(n_pairs);
    let n = c.num_qubits;
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let result = run_shots_compiled(&c, 10_000, 42).unwrap();

    for shot in &result.shots {
        for p in 0..n_pairs {
            assert_eq!(
                shot[2 * p],
                shot[2 * p + 1],
                "Bell pair {p} qubits disagree"
            );
        }
    }
}

#[test]
fn random_clifford_marginals() {
    let n = 10;
    let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let compiled = run_shots_compiled(&c, 50_000, 42).unwrap();
    let reference = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 50_000, 42).unwrap();

    for q in 0..n {
        let compiled_ones: usize = compiled.shots.iter().filter(|s| s[q]).count();
        let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
        let compiled_frac = compiled_ones as f64 / 50_000.0;
        let ref_frac = ref_ones as f64 / 50_000.0;
        assert!(
            (compiled_frac - ref_frac).abs() < 0.03,
            "Qubit {q} marginal mismatch: compiled={compiled_frac:.4} ref={ref_frac:.4}"
        );
    }
}

#[test]
fn lut_grouped_sampling_50q() {
    let n = 50;
    let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert!(sampler.rank() >= 40, "50q should have high rank for LUT");
    assert!(sampler.lut.is_some(), "rank >= 8 should build LUT");

    let compiled = run_shots_compiled(&c, 5_000, 42).unwrap();
    let reference = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 5_000, 42).unwrap();

    for q in 0..n {
        let compiled_ones: usize = compiled.shots.iter().filter(|s| s[q]).count();
        let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
        let compiled_frac = compiled_ones as f64 / 5_000.0;
        let ref_frac = ref_ones as f64 / 5_000.0;
        assert!(
            (compiled_frac - ref_frac).abs() < 0.05,
            "q{q} marginal mismatch: compiled={compiled_frac:.4} ref={ref_frac:.4}"
        );
    }
}

#[test]
fn forward_ghz_rank_and_distribution() {
    let n = 10;
    let mut c = circuits::ghz_circuit(n);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 1, "Forward GHZ-10 should have rank 1");

    let mut sampler = compile_forward(&c, 42).unwrap();
    let shots = sampler.sample_bulk(10_000);
    let all_zero: Vec<bool> = vec![false; n];
    let all_one: Vec<bool> = vec![true; n];
    let n_zero = shots.iter().filter(|s| *s == &all_zero).count();
    let n_one = shots.iter().filter(|s| *s == &all_one).count();
    assert_eq!(
        n_zero + n_one,
        10_000,
        "GHZ should produce only all-0 or all-1"
    );
    let ratio = n_zero as f64 / 10_000.0;
    assert!(
        (0.45..=0.55).contains(&ratio),
        "Expected ~50/50, got {ratio:.3}"
    );
}

#[test]
fn forward_bell_pairs_agree() {
    let n_pairs = 5;
    let mut c = circuits::independent_bell_pairs(n_pairs);
    let n = c.num_qubits;
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();
    assert_eq!(
        sampler.rank(),
        n_pairs,
        "Forward 5 Bell pairs should have rank 5"
    );

    let mut sampler = compile_forward(&c, 42).unwrap();
    let shots = sampler.sample_bulk(10_000);
    for shot in &shots {
        for p in 0..n_pairs {
            assert_eq!(
                shot[2 * p],
                shot[2 * p + 1],
                "Bell pair {p} qubits disagree"
            );
        }
    }
}

#[test]
fn forward_identity_all_zeros() {
    let mut c = Circuit::new(4, 4);
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 0, "Forward identity should have rank 0");

    let mut sampler = sampler;
    for _ in 0..100 {
        let outcome = sampler.sample();
        assert!(outcome.iter().all(|&b| !b), "All outcomes should be 0");
    }
}

#[test]
fn forward_single_h_measure() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    let sampler = compile_forward(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 1, "Forward H+measure should have rank 1");
}

#[test]
fn forward_x_measure_always_one() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let mut sampler = compile_forward(&c, 42).unwrap();
    assert_eq!(
        sampler.rank(),
        0,
        "X+measure should have rank 0 (deterministic)"
    );
    for _ in 0..100 {
        let outcome = sampler.sample();
        assert!(outcome[0], "X(q0) + measure should always give 1");
    }
}

#[test]
fn forward_random_clifford_marginals() {
    let n = 10;
    let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }

    let mut forward = compile_forward(&c, 42).unwrap();
    let mut backward = compile_measurements(&c, 42).unwrap();

    assert_eq!(forward.rank(), backward.rank(), "Ranks must match");

    let fwd_shots = forward.sample_bulk(50_000);
    let bwd_shots = backward.sample_bulk(50_000);

    for q in 0..n {
        let fwd_ones: usize = fwd_shots.iter().filter(|s| s[q]).count();
        let bwd_ones: usize = bwd_shots.iter().filter(|s| s[q]).count();
        let fwd_frac = fwd_ones as f64 / 50_000.0;
        let bwd_frac = bwd_ones as f64 / 50_000.0;
        assert!(
            (fwd_frac - bwd_frac).abs() < 0.03,
            "Qubit {q} marginal mismatch: forward={fwd_frac:.4} backward={bwd_frac:.4}"
        );
    }
}

#[test]
fn forward_clifford_50q_marginals() {
    let n = 50;
    let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }

    let mut forward = compile_forward(&c, 42).unwrap();
    let reference = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 5_000, 42).unwrap();

    let fwd_shots = forward.sample_bulk(5_000);

    for q in 0..n {
        let fwd_ones: usize = fwd_shots.iter().filter(|s| s[q]).count();
        let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
        let fwd_frac = fwd_ones as f64 / 5_000.0;
        let ref_frac = ref_ones as f64 / 5_000.0;
        assert!(
            (fwd_frac - ref_frac).abs() < 0.05,
            "q{q} marginal mismatch: forward={fwd_frac:.4} ref={ref_frac:.4}"
        );
    }
}

#[test]
fn rank_analysis_across_circuit_types() {
    let sizes = [10, 50, 100, 200];

    for &n in &sizes {
        let mut c = circuits::ghz_circuit(n);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(sampler.rank(), 1, "GHZ-{n} should have rank 1");
    }

    for &n in &sizes {
        let pairs = n / 2;
        let mut c = circuits::independent_bell_pairs(pairs);
        let nq = c.num_qubits;
        c.num_classical_bits = nq;
        for i in 0..nq {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(
            sampler.rank(),
            pairs,
            "Bell-{pairs} should have rank {pairs}"
        );
    }

    for &n in &sizes {
        let mut c = circuits::clifford_heavy_circuit(n, 50, 42);
        c.num_classical_bits = n;
        for i in 0..n {
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert!(
            sampler.rank() >= n - 1,
            "Random Clifford {n}q d50 should have rank ~{n}, got {}",
            sampler.rank()
        );
    }

    for &n in &sizes {
        let mut c = Circuit::new(n, n);
        for i in 0..n {
            c.add_gate(Gate::H, &[i]);
            c.add_measure(i, i);
        }
        let sampler = compile_measurements(&c, 42).unwrap();
        assert_eq!(
            sampler.rank(),
            n,
            "Product H-measure {n}q should have rank {n} (independent random bits)"
        );
    }
}

#[test]
fn filtered_bell_pairs_matches_monolithic() {
    let n_pairs = 50;
    let n = 2 * n_pairs;
    let mut c = circuits::independent_bell_pairs(n_pairs);
    c.num_classical_bits = n;
    for i in 0..n {
        c.add_measure(i, i);
    }

    let blocks = c.independent_subsystems();
    assert_eq!(
        blocks.len(),
        n_pairs,
        "Bell pairs should decompose into {n_pairs} blocks"
    );

    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(
        sampler.rank(),
        n_pairs,
        "Bell pairs rank should be {n_pairs}"
    );

    let shots = sampler.sample_bulk(10_000);
    for shot in &shots {
        for p in 0..n_pairs {
            assert_eq!(
                shot[2 * p],
                shot[2 * p + 1],
                "Bell pair {p}: qubits must agree"
            );
        }
    }

    let ones: usize = shots.iter().filter(|s| s[0]).count();
    let frac = ones as f64 / shots.len() as f64;
    assert!(
        (frac - 0.5).abs() < 0.05,
        "Bell pair first qubit should be ~50/50, got {frac:.3}"
    );
}

#[test]
fn filtered_product_h_matches_monolithic() {
    let n = 100;
    let mut c = Circuit::new(n, n);
    for i in 0..n {
        c.add_gate(Gate::H, &[i]);
        c.add_measure(i, i);
    }

    let blocks = c.independent_subsystems();
    assert_eq!(
        blocks.len(),
        n,
        "Product H should decompose into {n} blocks"
    );

    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), n);

    let shots = sampler.sample_bulk(5_000);
    for q in 0..n {
        let ones: usize = shots.iter().filter(|s| s[q]).count();
        let frac = ones as f64 / shots.len() as f64;
        assert!(
            (frac - 0.5).abs() < 0.06,
            "Qubit {q} should be ~50/50, got {frac:.3}"
        );
    }
}

#[test]
fn packed_shots_roundtrip_ghz() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(1000);
    assert_eq!(packed.num_shots(), 1000);
    assert_eq!(packed.num_measurements(), 10);

    let unpacked = packed.to_shots();
    assert_eq!(unpacked.len(), 1000);
    for shot in &unpacked {
        assert_eq!(shot.len(), 10);
        let first = shot[0];
        assert!(shot.iter().all(|&b| b == first));
    }
}

#[test]
fn packed_shots_matches_sample_bulk() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }

    let mut sampler1 = compile_forward(&c, 42).unwrap();
    let mut sampler2 = compile_forward(&c, 42).unwrap();

    let num_shots = 5000;
    let bulk = sampler1.sample_bulk(num_shots);
    let packed = sampler2.sample_bulk_packed(num_shots);
    let unpacked = packed.to_shots();

    assert_eq!(bulk.len(), unpacked.len());
    assert_eq!(bulk[0].len(), unpacked[0].len());

    let n = bulk[0].len();
    for q in 0..n {
        let freq1: usize = bulk.iter().filter(|s| s[q]).count();
        let freq2: usize = unpacked.iter().filter(|s| s[q]).count();
        let p1 = freq1 as f64 / num_shots as f64;
        let p2 = freq2 as f64 / num_shots as f64;
        assert!(
            (p1 - p2).abs() < 0.05,
            "qubit {q}: bulk={p1:.3}, packed={p2:.3}"
        );
    }
}

#[test]
fn packed_shots_get_bit() {
    let mut c = circuits::ghz_circuit(4);
    c.num_classical_bits = 4;
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(100);

    for s in 0..100 {
        let first = packed.get_bit(s, 0);
        for m in 1..4 {
            assert_eq!(packed.get_bit(s, m), first);
        }
    }
}

#[test]
fn packed_shots_counts() {
    let mut c = circuits::ghz_circuit(4);
    c.num_classical_bits = 4;
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(10_000);
    let counts = packed.counts();

    assert_eq!(counts.len(), 2);
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);
}

#[test]
fn sparse_parity_ghz() {
    let mut c = circuits::ghz_circuit(4);
    c.num_classical_bits = 4;
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();

    let sp = sampler.sparse().expect("sparse should be Some");
    assert_eq!(sp.num_rows, 4);

    let stats = sp.stats();
    assert!(stats.total_weight > 0);
    assert!(stats.min_weight <= stats.max_weight);

    for m in 0..4 {
        let cols = sp.row_cols(m);
        assert_eq!(cols.len(), sp.row_weight(m));
    }
}

#[test]
fn sparse_parity_matches_flip_rows() {
    let mut c = circuits::ghz_circuit(8);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();

    let sp = sampler.sparse().unwrap();
    let stats = sampler.parity_stats().unwrap();
    assert_eq!(stats.min_weight, sp.stats().min_weight);
    assert_eq!(stats.max_weight, sp.stats().max_weight);
}

#[test]
fn sparse_parity_empty_circuit() {
    let c = Circuit::new(2, 0);
    let sampler = compile_forward(&c, 42).unwrap();
    assert!(sampler.sparse().is_none());
}

#[test]
fn bts_meas_major_ghz_counts() {
    let mut c = circuits::ghz_circuit(8);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(10_000);

    assert_eq!(packed.layout(), ShotLayout::MeasMajor);

    let counts = packed.counts();
    assert_eq!(counts.len(), 2, "GHZ should have exactly 2 outcomes");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);
}

#[test]
fn bts_meas_major_get_bit_consistency() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(500);

    let shots = packed.to_shots();
    for (s, shot) in shots.iter().enumerate() {
        for (m, &val) in shot.iter().enumerate().take(20) {
            assert_eq!(packed.get_bit(s, m), val, "Mismatch at shot={s} meas={m}");
        }
    }
}

#[test]
fn bts_meas_major_marginals_match_stabilizer() {
    let mut c = circuits::clifford_heavy_circuit(50, 5, 42);
    c.num_classical_bits = 50;
    for i in 0..50 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(5_000);

    let reference = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 5_000, 42).unwrap();

    for q in 0..50 {
        let bts_ones: usize = (0..5_000).filter(|&s| packed.get_bit(s, q)).count();
        let ref_ones: usize = reference.shots.iter().filter(|s| s[q]).count();
        let bts_frac = bts_ones as f64 / 5_000.0;
        let ref_frac = ref_ones as f64 / 5_000.0;
        assert!(
            (bts_frac - ref_frac).abs() < 0.05,
            "q{q}: bts={bts_frac:.3} ref={ref_frac:.3}"
        );
    }
}

#[test]
fn streaming_counts_ghz() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let counts = sampler.sample_counts(10_000);

    assert_eq!(counts.len(), 2, "GHZ should produce exactly 2 outcomes");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);
}

#[test]
fn marginal_probabilities_ghz() {
    let mut c = circuits::ghz_circuit(8);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();
    let probs = sampler.marginal_probabilities();
    assert_eq!(probs.len(), 8);
    for &p in &probs {
        assert!((p - 0.5).abs() < 1e-10, "GHZ marginals should be 0.5");
    }
}

#[test]
fn marginal_probabilities_x_all_ones() {
    let mut c = Circuit::new(4, 4);
    for i in 0..4 {
        c.add_gate(Gate::X, &[i]);
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();
    let probs = sampler.marginal_probabilities();
    for &p in &probs {
        assert!(
            (p - 1.0).abs() < 1e-10,
            "X then measure should be deterministic 1"
        );
    }
}

#[test]
fn parity_report_not_empty() {
    let mut c = circuits::ghz_circuit(8);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_forward(&c, 42).unwrap();
    let report = sampler.parity_report();
    assert!(report.contains("measurements"));
    assert!(report.contains("rank"));
    assert!(report.contains("Weight"));
}

#[test]
fn weight_minimization_reduces_weight() {
    let mut rows: Vec<Vec<u64>> = vec![vec![0b1111], vec![0b1110], vec![0b1100]];
    let (before, after) = minimize_flip_row_weight(&mut rows);
    assert!(
        after <= before,
        "weight should not increase: {} -> {}",
        before,
        after
    );
    assert!(
        after < before,
        "weight should decrease for reducible rows: {} -> {}",
        before,
        after
    );
    assert_eq!(before, 4 + 3 + 2);
    assert_eq!(after, 1 + 1 + 2);
}

#[test]
fn weight_minimization_preserves_sampling() {
    let mut c = circuits::clifford_random_pairs(16, 20, 42);
    c.num_classical_bits = 16;
    for i in 0..16 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let counts = sampler.sample_bulk_packed(50_000).counts();
    let total: u64 = counts.values().sum();
    assert_eq!(total, 50_000);
    assert!(counts.len() > 1);
}

#[test]
fn xor_dag_reduces_weight() {
    let sp = SparseParity {
        col_indices: vec![0, 1, 0, 1, 2, 1, 2],
        row_offsets: vec![0, 2, 5, 7],
        num_rows: 3,
    };
    let dag = sp.build_xor_dag();
    assert!(
        dag.dag_weight < dag.original_weight,
        "DAG weight {} should be less than original {}",
        dag.dag_weight,
        dag.original_weight
    );
    assert_eq!(dag.original_weight, 2 + 3 + 2);
    assert!(dag.entries[1].parent.is_some() || dag.entries[2].parent.is_some());
}

#[test]
fn xor_dag_bts_correctness() {
    let mut c = circuits::clifford_random_pairs(16, 20, 42);
    c.num_classical_bits = 16;
    for i in 0..16 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(10_000);
    let counts = packed.counts();
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);
    assert!(counts.len() > 1);
}

#[test]
fn block_detection_independent_pairs() {
    let mut c = circuits::independent_bell_pairs(4);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    let sparse = sampler.sparse.as_ref().unwrap();
    let blocks = sparse.find_blocks(sampler.rank);
    assert!(blocks.is_some());
    let blocks = blocks.unwrap();
    assert_eq!(blocks.len(), 4);
    for b in &blocks {
        assert_eq!(b.len(), 2);
    }
}

#[test]
fn block_detection_single_block() {
    let mut c = circuits::clifford_random_pairs(8, 20, 42);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert!(sampler.parity_blocks.is_none());
}

#[test]
fn block_parallel_bts_correctness() {
    let mut c = circuits::independent_bell_pairs(8);
    c.num_classical_bits = 16;
    for i in 0..16 {
        c.add_measure(i, i);
    }

    let mut sampler_block = compile_measurements(&c, 42).unwrap();
    assert!(sampler_block.parity_blocks.is_some());

    let packed = sampler_block.sample_bulk_packed(10_000);
    let counts = packed.counts();
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);

    let shots = packed.to_shots();
    for shot in &shots {
        for pair in 0..8 {
            let b0 = shot[2 * pair];
            let b1 = shot[2 * pair + 1];
            assert_eq!(b0, b1, "Bell pair {pair} must be correlated");
        }
    }
}

#[test]
fn gray_code_exact_counts_bell_pair() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    let sampler = compile_measurements(&c, 42).unwrap();
    let counts = sampler.exact_counts().unwrap();
    let total: u64 = counts.values().sum();
    assert!(total.is_power_of_two());
    assert_eq!(counts.len(), 2);
    let half = total / 2;
    for &v in counts.values() {
        assert_eq!(v, half);
    }
}

#[test]
fn gray_code_exact_counts_ghz() {
    let mut c = Circuit::new(4, 4);
    c.add_gate(Gate::H, &[0]);
    for i in 0..3 {
        c.add_gate(Gate::Cx, &[i, i + 1]);
    }
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    let counts = sampler.exact_counts().unwrap();
    assert_eq!(counts.len(), 2);
    let total: u64 = counts.values().sum();
    let half = total / 2;
    for &v in counts.values() {
        assert_eq!(v, half);
    }
}

#[test]
fn gray_code_matches_sampling() {
    let mut c = circuits::clifford_random_pairs(8, 10, 42);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    let exact = sampler.exact_counts().unwrap();
    let total: u64 = exact.values().sum();
    let exact_probs: std::collections::HashMap<Vec<u64>, f64> = exact
        .iter()
        .map(|(k, &v)| (k.clone(), v as f64 / total as f64))
        .collect();

    let mut sampler2 = compile_measurements(&c, 123).unwrap();
    let packed = sampler2.sample_bulk_packed(100_000);
    let sample_counts = packed.counts();
    for (outcome, &exact_p) in &exact_probs {
        if exact_p > 0.01 {
            let sampled = *sample_counts.get(outcome).unwrap_or(&0) as f64 / 100_000.0;
            let diff = (sampled - exact_p).abs();
            assert!(
                diff < 0.02,
                "outcome {outcome:?}: exact={exact_p:.4}, sampled={sampled:.4}"
            );
        }
    }
}

#[test]
fn bts_batched_correctness() {
    let mut c = Circuit::new(4, 4);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    for i in 0..4 {
        c.add_measure(i, i);
    }

    let num_shots = BTS_BATCH_SHOTS * 3 + 100;
    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert!(sampler.should_use_bts(num_shots));

    let packed = sampler.sample_bulk_packed(num_shots);
    assert_eq!(packed.num_shots, num_shots);
    let counts = packed.counts();
    let total: u64 = counts.values().sum();
    assert_eq!(total, num_shots as u64);

    let shots = packed.to_shots();
    for shot in &shots {
        assert_eq!(shot[0], shot[1], "Bell pair 0 must be correlated");
        assert_eq!(shot[2], shot[3], "Bell pair 1 must be correlated");
    }
}

#[test]
fn memory_aware_bts_dispatch() {
    let mut c = circuits::clifford_random_pairs(100, 20, 42);
    c.num_classical_bits = 100;
    for i in 0..100 {
        c.add_measure(i, i);
    }
    let sampler = compile_measurements(&c, 42).unwrap();
    assert!(sampler.should_use_bts(100_000_000));
}

#[test]
fn detection_event_parity_matrix() {
    let mut c = Circuit::new(2, 4);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    c.add_measure(0, 2);
    c.add_measure(1, 3);

    let sampler = compile_measurements(&c, 42).unwrap();
    let sparse = sampler.sparse.as_ref().unwrap();

    let det = sparse.compile_detection_events(&[(2, 0), (3, 1)]);
    assert_eq!(det.num_rows, 2);
    for m in 0..2 {
        assert_eq!(
            det.row_weight(m),
            0,
            "same-stabilizer detection event must be deterministic"
        );
    }
}

#[test]
fn detection_event_sampling() {
    let mut c = Circuit::new(4, 8);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    for i in 0..4 {
        c.add_measure(i, i);
    }
    for i in 0..4 {
        c.add_measure(i, i + 4);
    }

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let packed = sampler.sample_detection_events(&[(4, 0), (5, 1), (6, 2), (7, 3)], 10_000);
    assert_eq!(packed.num_shots, 10_000);
    assert_eq!(packed.num_measurements, 4);

    let shots = packed.to_shots();
    for shot in &shots {
        for (i, &val) in shot.iter().enumerate().take(4) {
            assert!(!val, "detection event {i} must be 0");
        }
    }
}

#[test]
fn chunked_histogram_matches_direct() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }
    let num_shots = 100_000;

    let mut sampler_direct = compile_measurements(&c, 42).unwrap();
    let direct_counts = sampler_direct.sample_bulk_packed(num_shots).counts();

    let mut sampler_chunked = compile_measurements(&c, 42).unwrap();
    let chunked_counts = sampler_chunked.sample_counts(num_shots);

    assert_eq!(
        direct_counts, chunked_counts,
        "chunked histogram must match direct"
    );
}

#[test]
fn chunked_histogram_small_chunks() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let num_shots = 10_000;

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let mut acc = HistogramAccumulator::new();
    sampler.sample_chunked_with_size(num_shots, 300, &mut acc);
    let counts = acc.into_counts();

    let total: u64 = counts.values().sum();
    assert_eq!(total, num_shots as u64, "total shots must match");
    assert_eq!(counts.len(), 2, "GHZ-10 should have exactly 2 outcomes");

    let all_zeros = vec![0u64; 10usize.div_ceil(64)];
    let all_ones = {
        let mut v = vec![0u64; 10usize.div_ceil(64)];
        v[0] = (1u64 << 10) - 1;
        v
    };
    let c0 = counts.get(&all_zeros).copied().unwrap_or(0);
    let c1 = counts.get(&all_ones).copied().unwrap_or(0);
    assert!(
        (c0 as f64 / num_shots as f64 - 0.5).abs() < 0.03,
        "GHZ |0⟩^n fraction {c0}/{num_shots} should be ~0.5"
    );
    assert!(
        (c1 as f64 / num_shots as f64 - 0.5).abs() < 0.03,
        "GHZ |1⟩^n fraction {c1}/{num_shots} should be ~0.5"
    );
}

#[test]
fn marginals_ghz() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let marginals = sampler.sample_marginals(100_000);

    for (i, &p) in marginals.iter().enumerate() {
        assert!(
            (p - 0.5).abs() < 0.02,
            "GHZ qubit {i} marginal {p} should be ~0.5"
        );
    }
}

#[test]
fn marginals_accumulator_matches_direct() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }
    let num_shots = 50_000;

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(num_shots);
    let mut direct_ones = [0u64; 20];
    for s in 0..num_shots {
        for (m, count) in direct_ones.iter_mut().enumerate() {
            if packed.get_bit(s, m) {
                *count += 1;
            }
        }
    }
    let direct_marginals: Vec<f64> = direct_ones
        .iter()
        .map(|&c| c as f64 / num_shots as f64)
        .collect();

    let mut sampler2 = compile_measurements(&c, 42).unwrap();
    let chunked_marginals = sampler2.sample_marginals(num_shots);

    for (i, (d, ch)) in direct_marginals
        .iter()
        .zip(chunked_marginals.iter())
        .enumerate()
    {
        assert!(
            (d - ch).abs() < 1e-10,
            "marginal {i}: direct={d} chunked={ch}"
        );
    }
}

#[test]
fn pauli_expectation_ghz() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }

    let observables = vec![vec![0, 1], vec![0, 9], vec![3, 7]];

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let mut acc = PauliExpectationAccumulator::new(observables);
    sampler.sample_chunked(100_000, &mut acc);
    let exps = acc.expectations();

    for (i, &e) in exps.iter().enumerate() {
        assert!(
            (e - 1.0).abs() < 0.02,
            "GHZ Z_iZ_j observable {i} expectation {e} should be ~1.0"
        );
    }
}

#[test]
fn correlator_ghz() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }

    let pairs = vec![(0, 1), (0, 9), (3, 7)];
    let mut sampler = compile_measurements(&c, 42).unwrap();
    let mut acc = CorrelatorAccumulator::new(pairs);
    sampler.sample_chunked(100_000, &mut acc);
    let corrs = acc.correlators();

    for (i, &c) in corrs.iter().enumerate() {
        assert!(
            (c - 1.0).abs() < 0.02,
            "GHZ pair {i} correlator {c} should be ~1.0"
        );
    }
}

#[test]
fn marginals_non_64_aligned_shots() {
    let mut c = circuits::ghz_circuit(8);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let mut acc = MarginalsAccumulator::new(8);
    sampler.sample_chunked_with_size(137, 50, &mut acc);
    assert_eq!(acc.total_shots(), 137);

    let marginals = acc.marginals();
    for &p in &marginals {
        assert!(
            (p - 0.5).abs() < 0.15,
            "GHZ marginal {p} should be ~0.5 (small sample)"
        );
    }
}

#[test]
fn optimal_chunk_size_basic() {
    let cs = optimal_chunk_size(200, 256 * 1024 * 1024);
    assert!(cs >= 64);
    assert_eq!(cs % 64, 0);

    let cs_1000 = optimal_chunk_size(1000, 256 * 1024 * 1024);
    assert!(cs_1000 < cs, "more measurements should give smaller chunks");
    assert!(cs_1000 >= 64);
}

#[test]
fn noisy_chunked_histogram_matches_direct() {
    use crate::sim::noise::{compile_noisy, NoiseModel};

    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let noise = NoiseModel::uniform_depolarizing(&c, 0.001);
    let num_shots = 10_000;

    let mut sampler_direct = compile_noisy(&c, &noise, 42).unwrap();
    let (accum, m_words) = sampler_direct.sample_bulk_packed(num_shots);
    let mut direct_counts: std::collections::HashMap<Vec<u64>, u64> =
        std::collections::HashMap::new();
    for s in 0..num_shots {
        let shot = accum[s * m_words..s * m_words + m_words].to_vec();
        *direct_counts.entry(shot).or_insert(0) += 1;
    }

    let mut sampler_chunked = compile_noisy(&c, &noise, 42).unwrap();
    let chunked_counts = sampler_chunked.sample_counts(num_shots);

    assert_eq!(
        direct_counts, chunked_counts,
        "noisy chunked histogram must match direct"
    );
}
