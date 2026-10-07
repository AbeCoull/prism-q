use super::*;
use crate::circuit::{ClassicalCondition, Instruction, smallvec};
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
    c.measure_all();
    let sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), n_pairs, "5 Bell pairs should have rank 5");
}

#[test]
fn random_clifford_rank_is_n() {
    let n = 10;
    let mut c = circuits::clifford_heavy_circuit(n, 50, 42);
    c.measure_all();
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
fn reset_rejected() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_reset(0);
    c.add_measure(0, 0);
    let result = compile_measurements(&c, 42);
    assert!(result.is_err());
}

#[test]
fn defer_measure_reset_rewrites_reuse() {
    let mut c = Circuit::new(1, 2);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_reset(0);
    c.add_measure(0, 1);

    let deferred = defer_measure_reset_circuit(&c).unwrap();
    assert_eq!(deferred.num_qubits, 2);
    assert!(!deferred.has_resets());
    assert!(deferred.has_terminal_measurements_only());
    assert_eq!(deferred.measurement_map(), vec![(0, 0), (1, 1)]);

    let mut sampler = compile_measurements(&deferred, 42).unwrap();
    for shot in sampler.sample_bulk(64) {
        assert!(shot[0]);
        assert!(!shot[1]);
    }
}

#[test]
fn defer_measure_reset_rejects_measured_qubit_reuse() {
    let mut c = Circuit::new(1, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 1);

    let result = defer_measure_reset_circuit(&c);
    assert!(result.is_err());
}

#[test]
fn run_shots_deferred_reset_preserves_bell_correlation() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(1, 0);
    c.add_reset(1);
    c.add_measure(0, 1);

    let result = crate::sim::run_shots_with(BackendKind::Stabilizer, &c, 1024, 42).unwrap();
    assert_eq!(result.num_classical_bits, 2);
    let mut saw_zero = false;
    let mut saw_one = false;
    for shot in &result.shots {
        assert_eq!(shot[0], shot[1]);
        saw_zero |= !shot[0];
        saw_one |= shot[0];
    }
    assert!(saw_zero);
    assert!(saw_one);
}

#[test]
fn conditional_rejected() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: smallvec![1],
    });
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
    c.measure_all();
    let result = run_shots_compiled(&c, 10_000, 42).unwrap();
    let counts = result.counts();

    let key_zero = vec![0u64];
    let key_one = vec![(1u64 << n) - 1];

    let n_zero = counts.get(&key_zero).copied().unwrap_or(0);
    let n_one = counts.get(&key_one).copied().unwrap_or(0);

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
    c.measure_all();
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
fn run_shots_compiled_follows_the_measurement_map() {
    let mut c = Circuit::new(3, 5);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 1]);
    c.add_measure(0, 4);
    c.add_measure(1, 0);
    c.add_measure(2, 2);
    c.add_measure(0, 0);

    let result = run_shots_compiled(&c, 256, 42).unwrap();
    assert!(result.shots.iter().all(|shot| shot.len() == 5));
    let counts = result.counts();
    assert_eq!(counts.values().sum::<u64>(), 256);
    for key in counts.keys() {
        assert!(
            *key == vec![0b10001] || *key == vec![0b10101],
            "unexpected key {key:?}"
        );
    }
    assert_eq!(counts.len(), 2);
}

#[test]
fn random_clifford_marginals() {
    let n = 10;
    let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
    c.measure_all();
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
    c.measure_all();
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
    c.measure_all();
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
    c.measure_all();
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
    c.measure_all();

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
    c.measure_all();

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

fn reduce_gf2(mut v: Vec<u64>, basis: &[(usize, Vec<u64>)]) -> Vec<u64> {
    for (pivot, row) in basis {
        if get_bit(&v, *pivot) {
            xor_words(&mut v, row);
        }
    }
    v
}

fn lowest_set_bit(v: &[u64]) -> Option<usize> {
    v.iter()
        .position(|&w| w != 0)
        .map(|i| i * 64 + v[i].trailing_zeros() as usize)
}

fn measure_in_order(mut c: Circuit, qubits: &[usize]) -> Circuit {
    c.num_classical_bits = qubits.len();
    for (bit, &q) in qubits.iter().enumerate() {
        c.add_measure(q, bit);
    }
    c
}

// Every outcome of an independent tableau run must lie in the compiled affine space,
// and enough runs span a space of the compiled rank, which pins the space exactly.
fn assert_forward_matches_tableau(c: &Circuit, runs: u64) {
    use crate::backend::Backend;
    use crate::backend::stabilizer::StabilizerBackend;

    let sampler = compile_forward(c, 42).unwrap();
    let mut compiled: Vec<(usize, Vec<u64>)> = Vec::new();
    for row in &sampler.flip_rows {
        let r = reduce_gf2(row.clone(), &compiled);
        let pivot = lowest_set_bit(&r).expect("flip rows are independent");
        compiled.push((pivot, r));
    }
    assert_eq!(compiled.len(), sampler.rank());

    let mut observed: Vec<(usize, Vec<u64>)> = Vec::new();
    for seed in 0..runs {
        let mut b = StabilizerBackend::new(seed);
        b.init(c.num_qubits, c.num_classical_bits).unwrap();
        for inst in &c.instructions {
            b.apply(inst).unwrap();
        }
        let mut diff = pack_bools(b.classical_results());
        xor_words(&mut diff, sampler.ref_bits_packed());
        assert!(
            lowest_set_bit(&reduce_gf2(diff.clone(), &compiled)).is_none(),
            "run {seed} lies outside the compiled space"
        );
        let r = reduce_gf2(diff, &observed);
        if let Some(pivot) = lowest_set_bit(&r) {
            observed.push((pivot, r));
        }
    }
    assert_eq!(observed.len(), sampler.rank(), "runs span a smaller space");
}

#[test]
fn forward_matches_tableau_across_words_in_scrambled_order() {
    let n = 130;
    let order: Vec<usize> = (0..n).map(|i| (i * 67) % n).collect();
    let c = measure_in_order(circuits::clifford_heavy_circuit(n, 2, 7), &order);
    assert_forward_matches_tableau(&c, 80);
}

#[test]
fn forward_matches_tableau_on_a_scrambled_ghz() {
    let n = 200;
    let order: Vec<usize> = (0..n).map(|i| (i * 77) % n).collect();
    let c = measure_in_order(circuits::ghz_circuit(n), &order);
    assert_forward_matches_tableau(&c, 20);
}

#[test]
fn forward_matches_tableau_measuring_in_register_order() {
    let n = 140;
    let order: Vec<usize> = (0..n).collect();
    let c = measure_in_order(circuits::clifford_heavy_circuit(n, 2, 5), &order);
    assert_forward_matches_tableau(&c, 80);
}

#[test]
fn forward_matches_tableau_on_a_partial_register() {
    let n = 150;
    let order: Vec<usize> = (0..n).rev().step_by(3).collect();
    let c = measure_in_order(circuits::clifford_heavy_circuit(n, 3, 11), &order);
    assert_forward_matches_tableau(&c, 120);
}

#[test]
fn rank_analysis_across_circuit_types() {
    let sizes = [10, 50, 100, 200];

    for &n in &sizes {
        let mut c = circuits::ghz_circuit(n);
        c.measure_all();
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
        c.measure_all();
        let sampler = compile_measurements(&c, 42).unwrap();
        assert!(
            sampler.rank() >= n.saturating_sub(4),
            "Random Clifford {n}q d50 should have high rank, got {}",
            sampler.rank()
        );
    }

    for &n in &sizes {
        let mut c = Circuit::new(n, n);
        for i in 0..n {
            c.add_gate(Gate::H, &[i]);
        }
        for i in 0..n {
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
    let mut c = circuits::independent_bell_pairs(n_pairs);
    c.measure_all();

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
fn filtered_blocks_sharing_classical_bits() {
    let mut c = Circuit::new(4, 2);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    c.add_measure(2, 0);
    c.add_measure(3, 1);

    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 0);
    for shot in sampler.sample_bulk(8) {
        assert_eq!(shot, vec![false, false, true, true]);
    }
}

#[test]
fn filtered_product_h_matches_monolithic() {
    let n = 100;
    let mut c = Circuit::new(n, n);
    for i in 0..n {
        c.add_gate(Gate::H, &[i]);
    }
    for i in 0..n {
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
fn packed_shots_parity_rows_shot_major() {
    let packed = PackedShots::from_shot_major(vec![0b001, 0b011, 0b110], 3, 3);
    let rows = vec![vec![0, 1], vec![1, 2], Vec::new()];
    let parity = packed.parity_rows(&rows).unwrap();
    let shots = parity.to_shots();
    assert_eq!(
        shots,
        vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![true, false, false],
        ]
    );
}

#[test]
fn packed_shots_parity_rows_meas_major() {
    let packed = PackedShots::from_meas_major(vec![0b011, 0b110, 0b100], 3, 3);
    let rows = vec![vec![0, 1], vec![1, 2], Vec::new()];
    let parity = packed.parity_rows(&rows).unwrap();
    assert_eq!(parity.layout(), ShotLayout::MeasMajor);
    assert_eq!(
        parity.to_shots(),
        vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![true, false, false],
        ]
    );
}

#[test]
fn packed_shots_parity_rows_rejects_bad_index() {
    let packed = PackedShots::from_shot_major(vec![0], 1, 1);
    assert!(packed.parity_rows(&[vec![1]]).is_err());
}

#[test]
fn packed_shots_try_constructors_reject_invalid_raw_data() {
    for (name, result) in [
        (
            "shot-major shape",
            PackedShots::try_from_shot_major(vec![0], 2, 65),
        ),
        (
            "measurement-major shape",
            PackedShots::try_from_meas_major(vec![0], 65, 2),
        ),
        (
            "shot-major tail",
            PackedShots::try_from_shot_major(vec![0b1000], 1, 3),
        ),
        (
            "measurement-major tail",
            PackedShots::try_from_meas_major(vec![0b1000], 3, 1),
        ),
    ] {
        assert!(
            matches!(result.unwrap_err(), PrismError::InvalidParameter { .. }),
            "{name}"
        );
    }
}

#[test]
fn packed_shots_reports_raw_format_version() {
    let packed = PackedShots::from_shot_major(vec![0], 1, 1);
    assert_eq!(packed.raw_format_version(), PackedShots::RAW_FORMAT_VERSION);
    assert_eq!(PackedShots::RAW_FORMAT_VERSION, 1);
}

#[test]
fn packed_shots_counts_ignore_shot_major_padding() {
    let packed = PackedShots {
        data: vec![0b1000, 0b1001, 0b0000],
        num_shots: 3,
        num_measurements: 3,
        m_words: 1,
        s_words: 1,
        layout: ShotLayout::ShotMajor,
    };

    let counts = packed.counts();
    assert_eq!(counts.get(&vec![0]), Some(&2));
    assert_eq!(counts.get(&vec![1]), Some(&1));

    let mut acc = HistogramAccumulator::new();
    acc.accumulate(&packed);
    let acc_counts = acc.into_counts();
    assert_eq!(acc_counts.get(&vec![0]), Some(&2));
    assert_eq!(acc_counts.get(&vec![1]), Some(&1));
}

#[test]
fn packed_shots_marginals_ignore_meas_major_padding() {
    let packed = PackedShots {
        data: vec![0b1001],
        num_shots: 3,
        num_measurements: 1,
        m_words: 1,
        s_words: 1,
        layout: ShotLayout::MeasMajor,
    };

    let mut acc = MarginalsAccumulator::new(1);
    acc.accumulate(&packed);
    assert!((acc.marginals()[0] - (1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn packed_shots_parity_rows_clear_meas_major_padding() {
    let packed = PackedShots {
        data: vec![0b1001, 0b1000],
        num_shots: 3,
        num_measurements: 2,
        m_words: 1,
        s_words: 1,
        layout: ShotLayout::MeasMajor,
    };

    let parity = packed.parity_rows(&[vec![0], vec![0, 1]]).unwrap();
    assert_eq!(parity.raw_data(), &[0b001, 0b001]);
}

#[test]
fn detector_sampler_preserves_bell_measure_reset_correlation() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(1, 0);
    c.add_reset(1);
    c.add_measure(0, 1);

    let detector_rows = vec![vec![0, 1], vec![0]];
    let observable_rows = vec![vec![1]];
    let mut sampler = compile_detector_sampler(&c, detector_rows, observable_rows, 42).unwrap();
    assert_eq!(sampler.num_measurements(), 2);
    assert_eq!(sampler.num_detectors(), 2);
    assert_eq!(sampler.num_observables(), 1);

    let batch = sampler.sample_packed(1024).unwrap();
    assert_eq!(batch.detectors.num_measurements(), 2);
    assert_eq!(batch.observables.num_measurements(), 1);

    let detector_shots = batch.detectors.to_shots();
    let observable_shots = batch.observables.to_shots();
    let measurement_shots = batch.measurements.to_shots();
    let mut saw_zero = false;
    let mut saw_one = false;
    for shot_idx in 0..detector_shots.len() {
        assert!(!detector_shots[shot_idx][0]);
        assert_eq!(detector_shots[shot_idx][1], measurement_shots[shot_idx][0]);
        assert_eq!(
            observable_shots[shot_idx][0],
            measurement_shots[shot_idx][1]
        );
        saw_zero |= !detector_shots[shot_idx][1];
        saw_one |= detector_shots[shot_idx][1];
    }
    assert!(saw_zero);
    assert!(saw_one);
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
fn packed_shots_counts_support_wide_outputs() {
    let n = 600;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        c.add_measure(q, q);
    }

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(256);
    let counts = packed.counts();
    let total: u64 = counts.values().sum();
    assert_eq!(total, 256);
    assert!(counts.keys().all(|key| key.len() == n.div_ceil(64)));
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
fn bts_bulk_sampling_conserves_shots() {
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
fn compiled_sampler_scale_fixture_reaches_batched_bts() {
    let mut circuit = circuits::clifford_heavy_circuit(100, 10, 0xDEAD_BEEF);
    circuit.num_classical_bits = 100;
    for qubit in 0..100 {
        circuit.add_measure(qubit, qubit);
    }
    let sampler = compile_forward(&circuit, 0xDEAD_BEEF).unwrap();
    assert!(sampler.parity_blocks.is_none());
    for shots in [100_000, 1_000_000, 10_000_000] {
        assert!(shots > BTS_BATCH_SHOTS);
        assert!(sampler.should_use_bts(shots));
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
fn detection_events_carry_reference_bits_on_every_path() {
    // q8 holds the parity of eight random bits, q9 copies q0, q10 is flipped
    // and q11 stays at zero.
    let build = |random: bool| {
        let mut c = Circuit::new(12, 12);
        for q in 0..8 {
            if random {
                c.add_gate(Gate::H, &[q]);
            }
            c.add_gate(Gate::Cx, &[q, 8]);
        }
        c.add_gate(Gate::Cx, &[0, 9]);
        c.add_gate(Gate::X, &[10]);
        for q in 0..12 {
            c.add_measure(q, q);
        }
        c
    };
    let base = [(0, 9), (10, 11), (11, 10), (8, 11)];
    // Repeating the weight-8 pair pushes the projected parity weight past the
    // measurement weight, which selects the path that samples records first.
    let repeated: Vec<(usize, usize)> = base.iter().copied().cycle().take(64).collect();

    for (random, pairs) in [(true, &base[..]), (true, &repeated[..]), (false, &base[..])] {
        let mut sampler = compile_measurements(&build(random), 42).unwrap();
        let num_shots = 1000;
        let shots = sampler.sample_detection_events(pairs, num_shots).to_shots();
        assert_eq!(shots.len(), num_shots);
        let (mut fired, mut trials) = (0, 0);
        for shot in &shots {
            for (e, &(a, b)) in pairs.iter().enumerate() {
                match (a.min(b), a.max(b)) {
                    (0, 9) => assert!(!shot[e], "copied pair event {e} fired"),
                    (10, 11) => assert!(shot[e], "flipped pair event {e} did not fire"),
                    _ => {
                        fired += usize::from(shot[e]);
                        trials += 1;
                    }
                }
            }
        }
        if random {
            assert!(
                fired > trials / 3 && fired < 2 * trials / 3,
                "parity event fired {fired} of {trials}"
            );
        } else {
            assert_eq!(fired, 0);
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
    let num_shots = 50_000;

    let mut sampler_direct = compile_measurements(&c, 42).unwrap();
    let direct_counts = sampler_direct.sample_bulk_packed(num_shots).counts();

    let mut sampler_chunked = compile_measurements(&c, 42).unwrap();
    let mut acc = super::accumulator::HistogramAccumulator::new();
    sampler_chunked.sample_chunked(num_shots, &mut acc);
    let chunked_counts = acc.into_counts();

    assert_eq!(
        direct_counts, chunked_counts,
        "chunked histogram must match direct"
    );
}

#[test]
fn rank_space_counts_valid() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }
    let num_shots = 100_000;

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let rank = sampler.rank();
    assert!(
        rank <= 20,
        "rank {rank} should be ≤20 to use rank-space counting"
    );
    let counts = sampler.sample_counts(num_shots);

    let total: u64 = counts.values().sum();
    assert_eq!(total, num_shots as u64, "total shots must match");
    assert!(
        counts.len() <= (1 << rank),
        "distinct outcomes {} must be ≤ 2^rank={}",
        counts.len(),
        1 << rank
    );

    let m_words = 20usize.div_ceil(64);
    for key in counts.keys() {
        assert_eq!(
            key.len(),
            m_words,
            "each key must have m_words={m_words} words"
        );
    }
}

#[test]
fn rank_space_counts_matches_distribution() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let num_shots = 100_000;

    let mut sampler_rs = compile_measurements(&c, 42).unwrap();
    let rs_counts = sampler_rs.sample_counts(num_shots);

    let mut sampler_old = compile_measurements(&c, 99).unwrap();
    let mut acc = super::accumulator::HistogramAccumulator::new();
    sampler_old.sample_chunked(num_shots, &mut acc);
    let old_counts = acc.into_counts();

    assert_eq!(
        rs_counts.keys().collect::<std::collections::HashSet<_>>(),
        old_counts.keys().collect::<std::collections::HashSet<_>>(),
        "rank-space and chunked must produce same outcome set for GHZ"
    );

    let total_rs: u64 = rs_counts.values().sum();
    let total_old: u64 = old_counts.values().sum();
    assert_eq!(total_rs, num_shots as u64);
    assert_eq!(total_old, num_shots as u64);

    for (key, &rs_count) in &rs_counts {
        let old_count = old_counts[key];
        let expected = num_shots as f64 / rs_counts.len() as f64;
        let diff = (rs_count as f64 - old_count as f64).abs();
        assert!(
            diff < expected * 0.1,
            "counts for {key:?} differ too much: rs={rs_count} old={old_count}"
        );
    }
}

fn deterministic_circuit(clbit_of: impl Fn(usize) -> usize) -> Circuit {
    let n = 70;
    let num_clbits = 75;
    let mut c = Circuit::new(n, num_clbits);
    for q in (0..n).step_by(3) {
        c.add_gate(Gate::X, &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        c.add_measure(q, clbit_of(q) % num_clbits);
    }
    c
}

fn chunked_counts(
    sampler: &mut CompiledSampler,
    shots: usize,
) -> std::collections::HashMap<Vec<u64>, u64> {
    counts_from_chunks(|acc| sampler.sample_chunked(shots, acc))
}

#[test]
fn rank_zero_counts_match_chunked_histogram() {
    let c = deterministic_circuit(|q| q * 11 + 3);
    for mut sampler in [
        compile_measurements(&c, 42).unwrap(),
        compile_forward(&c, 42).unwrap(),
    ] {
        assert_eq!(sampler.rank(), 0);
        for shots in [0, 1, 1000, 100_000] {
            let expected = chunked_counts(&mut sampler, shots);
            let first = sampler.sample_counts(shots);
            let second = sampler.sample_counts(shots);
            assert_eq!(first, expected, "shots={shots}");
            assert_eq!(second, expected, "shots={shots}");
        }
        assert!(sampler.sample_counts(0).is_empty());
    }
}

#[test]
fn rank_zero_counts_without_measurements() {
    let c = circuits::ghz_circuit(5);
    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 0);
    for shots in [0, 7] {
        assert_eq!(
            sampler.sample_counts(shots),
            chunked_counts(&mut sampler, shots)
        );
    }
}

#[test]
fn rank_zero_run_counts_match_shot_counts() {
    let c = deterministic_circuit(|q| q);
    let counts = crate::sim::run_counts_with(BackendKind::Auto, &c, 5000, 42)
        .unwrap()
        .0;
    let shots = crate::sim::run_shots_with(BackendKind::Auto, &c, 5000, 42).unwrap();
    assert_eq!(counts.len(), 1);
    assert_eq!(counts, shots.counts());
}

#[test]
fn accumulator_matches_packed_counts() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(100_000);
    let direct_counts = packed.counts();

    let mut acc = super::accumulator::HistogramAccumulator::new();
    acc.accumulate(&packed);
    let acc_counts = acc.into_counts();

    assert_eq!(
        direct_counts, acc_counts,
        "accumulator counts must match PackedShots::counts"
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
    use crate::sim::noise::{NoiseModel, compile_noisy};

    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let noise = NoiseModel::uniform_depolarizing(&c, 0.001);
    let num_shots = 10_000;

    let mut sampler_direct = compile_noisy(&c, &noise, 42).unwrap();
    let packed = sampler_direct.sample_bulk_packed(num_shots);
    let direct_counts = packed.counts();

    let mut sampler_chunked = compile_noisy(&c, &noise, 42).unwrap();
    let chunked_counts = sampler_chunked.sample_counts(num_shots);

    assert_eq!(
        direct_counts, chunked_counts,
        "noisy chunked histogram must match direct"
    );
}

#[test]
fn multinomial_ghz_correct_outcomes() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 1);

    let counts = sampler.sample_counts(100_000);
    assert_eq!(counts.len(), 2, "GHZ should produce exactly 2 outcomes");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 100_000);

    for (key, &count) in &counts {
        let frac = count as f64 / 100_000.0;
        assert!(
            (frac - 0.5).abs() < 0.02,
            "GHZ outcome {key:?} fraction {frac:.4} should be ~0.5"
        );
    }
}

#[test]
fn multinomial_bell_pairs_correlation() {
    let n_pairs = 5;
    let mut c = circuits::independent_bell_pairs(n_pairs);
    c.measure_all();
    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), n_pairs);

    let counts = sampler.sample_counts(100_000);
    let total: u64 = counts.values().sum();
    assert_eq!(total, 100_000);
    assert_eq!(
        counts.len(),
        1 << n_pairs,
        "5 Bell pairs should have 2^5=32 outcomes"
    );

    for key in counts.keys() {
        for p in 0..n_pairs {
            let b0 = (key[0] >> (2 * p)) & 1;
            let b1 = (key[0] >> (2 * p + 1)) & 1;
            assert_eq!(
                b0, b1,
                "Bell pair {p} must be correlated in outcome {key:?}"
            );
        }
    }
}

#[test]
fn multinomial_matches_exact_distribution() {
    let mut c = circuits::clifford_random_pairs(8, 10, 42);
    c.num_classical_bits = 8;
    for i in 0..8 {
        c.add_measure(i, i);
    }

    let sampler_exact = compile_measurements(&c, 42).unwrap();
    let exact = sampler_exact.exact_counts().unwrap();
    let exact_total: u64 = exact.values().sum();
    let exact_probs: std::collections::HashMap<Vec<u64>, f64> = exact
        .iter()
        .map(|(k, &v)| (k.clone(), v as f64 / exact_total as f64))
        .collect();

    let num_shots = 200_000;
    let mut sampler_multi = compile_measurements(&c, 42).unwrap();
    let rank = sampler_multi.rank();
    assert!(rank <= 8);
    let multi_counts = sampler_multi.sample_counts(num_shots);

    let total: u64 = multi_counts.values().sum();
    assert_eq!(total, num_shots as u64);

    for (key, &exact_p) in &exact_probs {
        if exact_p > 0.005 {
            let sampled = *multi_counts.get(key).unwrap_or(&0) as f64 / num_shots as f64;
            assert!(
                (sampled - exact_p).abs() < 0.02,
                "outcome {key:?}: exact={exact_p:.4} multinomial={sampled:.4}"
            );
        }
    }
}

#[test]
fn multinomial_identity_all_zeros() {
    let mut c = Circuit::new(4, 4);
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert_eq!(sampler.rank(), 0);

    let counts = sampler.sample_counts(10_000);
    assert_eq!(counts.len(), 1, "Identity circuit should produce 1 outcome");
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);
}

#[test]
fn multinomial_high_rank_falls_through() {
    let n = 50;
    let mut c = circuits::clifford_heavy_circuit(n, 10, 42);
    c.measure_all();
    let mut sampler = compile_measurements(&c, 42).unwrap();
    assert!(sampler.rank() > 22, "50q clifford should have rank > 22");

    let counts = sampler.sample_counts(10_000);
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);
    assert!(counts.len() > 1);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_bts_below_threshold_with_stub_context_stays_on_cpu() {
    let threshold = crate::gpu::bts_min_shots();
    if threshold <= 1 {
        return;
    }

    let n = 128;
    let shots = threshold.min(1024).saturating_sub(1).max(1);
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        c.add_measure(q, q);
    }

    let mut cpu = compile_measurements(&c, 42).unwrap();
    assert!(cpu.should_use_bts(shots));
    assert!(cpu.parity_blocks.is_none());
    let cpu_counts = cpu.sample_bulk_packed(shots).counts();

    let mut gpu = compile_measurements(&c, 42)
        .unwrap()
        .with_gpu(crate::gpu::GpuContext::stub_for_tests());
    assert!(gpu.should_use_bts(shots));
    assert!(!gpu.should_use_gpu_bts(shots));
    let gpu_counts = gpu.sample_bulk_packed(shots).counts();

    assert!(gpu.gpu_bts_cache.is_none());
    assert_eq!(cpu_counts, gpu_counts);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_bts_low_rank_ghz_stays_on_cpu_above_threshold() {
    let shots = crate::gpu::bts_min_shots().max(1);
    let n = 128;
    let mut c = circuits::ghz_circuit(n);
    c.measure_all();

    let gpu = compile_measurements(&c, 42)
        .unwrap()
        .with_gpu(crate::gpu::GpuContext::stub_for_tests());
    assert_eq!(gpu.rank(), 1, "GHZ should compile to rank 1");
    assert!(!gpu.should_use_gpu_bts(shots));
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_bts_low_weight_h_layer_stays_on_cpu_above_threshold() {
    let shots = crate::gpu::bts_min_shots().max(1);
    let n = 128;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        c.add_measure(q, q);
    }

    let gpu = compile_measurements(&c, 42)
        .unwrap()
        .with_gpu(crate::gpu::GpuContext::stub_for_tests());
    assert!(gpu.should_use_bts(shots));
    assert_eq!(gpu.rank(), n, "independent H layer should have full rank");
    assert!(!gpu.should_use_gpu_bts(shots));
}

fn key_bit(key: &[u64], bit: usize) -> bool {
    key[bit / 64] >> (bit % 64) & 1 == 1
}

fn assert_padding_clear(key: &[u64], num_bits: usize) {
    for bit in num_bits..key.len() * 64 {
        assert!(!key_bit(key, bit), "padding bit {bit} set");
    }
}

#[test]
fn independent_counts_cross_the_word_rank() {
    for rank in [63usize, 64, 65, 128] {
        let one = rank;
        let zero = rank + 1;
        let mut c = Circuit::new(rank + 2, rank + 2);
        for q in 0..rank {
            c.add_gate(Gate::H, &[q]);
        }
        c.add_gate(Gate::X, &[one]);
        for q in 0..rank + 2 {
            c.add_measure(q, q);
        }

        for shots in [0usize, 1000] {
            let mut sampler = compile_measurements(&c, 42).unwrap();
            assert_eq!(sampler.rank(), rank);
            let counts = sampler.sample_counts(shots);
            assert_eq!(counts.values().sum::<u64>(), shots as u64, "rank {rank}");
            assert_eq!(
                counts.len(),
                shots,
                "rank {rank}: draws over 2^{rank} collided"
            );
            for key in counts.keys() {
                assert_eq!(key.len(), (rank + 2).div_ceil(64));
                assert!(key_bit(key, one) && !key_bit(key, zero), "rank {rank}");
                assert_padding_clear(key, rank + 2);
            }
        }
    }
}

#[test]
fn wide_rank_counts_follow_the_measurement_map() {
    for rank in [63usize, 64, 65, 128] {
        let one = rank;
        let zero = rank + 1;
        let num_bits = rank + 4;
        let (twice, unused, wide) = (rank, rank + 2, rank + 3);
        let mut c = Circuit::new(rank + 2, num_bits);
        for q in 0..rank {
            c.add_gate(Gate::H, &[q]);
        }
        c.add_gate(Gate::X, &[one]);
        for q in 0..rank {
            c.add_measure(q, rank - 1 - q);
        }
        c.add_measure(zero, twice);
        c.add_measure(one, twice);
        c.add_measure(one, wide);
        c.add_measure(zero, rank + 1);

        for shots in [0usize, 1000] {
            let counts = crate::sim::simulate(&c)
                .seed(42)
                .sample_counts(shots)
                .unwrap()
                .counts;
            assert_eq!(counts.values().sum::<u64>(), shots as u64, "rank {rank}");
            assert_eq!(
                counts.len(),
                shots,
                "rank {rank}: draws over 2^{rank} collided"
            );
            for key in counts.keys() {
                assert_eq!(key.len(), num_bits.div_ceil(64));
                assert!(key_bit(key, twice), "rank {rank}: last write wins");
                assert!(key_bit(key, wide) && !key_bit(key, unused), "rank {rank}");
                assert!(!key_bit(key, rank + 1), "rank {rank}");
                assert_padding_clear(key, num_bits);
            }
        }
    }
}

// Rotated surface-code memory in measure-and-reset form: an ancilla per plaquette,
// `rounds` syndrome rounds, then a Z readout of every data qubit.
fn surface_memory_mr(distance: usize, rounds: usize) -> Circuit {
    let mut plaquettes: Vec<(bool, Vec<usize>)> = Vec::new();
    for i in 0..=distance {
        for j in 0..=distance {
            let cells: Vec<usize> = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
                .into_iter()
                .filter(|&(r, c)| (1..=distance).contains(&r) && (1..=distance).contains(&c))
                .map(|(r, c)| (r - 1) * distance + (c - 1))
                .collect();
            let z_type = (i + j) % 2 == 1;
            let boundary_kept = if z_type {
                i == 0 || i == distance
            } else {
                j == 0 || j == distance
            };
            if cells.len() == 4 || (cells.len() == 2 && boundary_kept) {
                plaquettes.push((z_type, cells));
            }
        }
    }
    let data = distance * distance;
    let n = data + plaquettes.len();
    let mut c = Circuit::new(n, rounds * plaquettes.len() + data);
    let mut bit = 0;
    for _ in 0..rounds {
        for (a, (z_type, cells)) in plaquettes.iter().enumerate() {
            let anc = data + a;
            if !z_type {
                c.add_gate(Gate::H, &[anc]);
            }
            for &q in cells {
                if *z_type {
                    c.add_gate(Gate::Cx, &[q, anc]);
                } else {
                    c.add_gate(Gate::Cx, &[anc, q]);
                }
            }
            if !z_type {
                c.add_gate(Gate::H, &[anc]);
            }
            c.add_measure(anc, bit);
            c.add_reset(anc);
            bit += 1;
        }
    }
    for q in 0..data {
        c.add_measure(q, bit);
        bit += 1;
    }
    c
}

fn repetition_memory_mr(distance: usize, rounds: usize) -> Circuit {
    let n = 2 * distance - 1;
    let mut c = Circuit::new(n, rounds * (distance - 1) + distance);
    let mut bit = 0;
    for _ in 0..rounds {
        for a in 0..distance - 1 {
            let anc = distance + a;
            c.add_gate(Gate::Cx, &[a, anc]);
            c.add_gate(Gate::Cx, &[a + 1, anc]);
            c.add_measure(anc, bit);
            c.add_reset(anc);
            bit += 1;
        }
    }
    for q in 0..distance {
        c.add_measure(q, bit);
        bit += 1;
    }
    c
}

fn assert_live_matches_dense(c: &Circuit, label: &str) {
    let schedule = live_forward::LiveSchedule::plan(c);
    let live = live_forward::compile_forward_live(c, 42, &schedule).unwrap();
    let dense = compile_forward_dense(c, 42).unwrap();
    assert_eq!(live.rank(), dense.rank(), "{label}: rank");
    assert_eq!(
        live.ref_bits_packed(),
        dense.ref_bits_packed(),
        "{label}: reference bits"
    );
    assert_eq!(live.flip_rows, dense.flip_rows, "{label}: flip rows");
    let dispatched = compile_forward(c, 42).unwrap();
    assert_eq!(dispatched.flip_rows, dense.flip_rows, "{label}: dispatch");
}

#[test]
fn live_forward_matches_dense_on_deferred_memories() {
    for (d, r) in [(3, 12), (5, 6), (7, 3)] {
        let c = defer_measure_reset_circuit(&surface_memory_mr(d, r)).unwrap();
        assert!(c.num_qubits >= 64);
        assert_live_matches_dense(&c, &format!("surface d{d} r{r}"));
    }
    for (d, r) in [(15, 8), (9, 20)] {
        let c = defer_measure_reset_circuit(&repetition_memory_mr(d, r)).unwrap();
        assert!(c.num_qubits >= 64);
        assert_live_matches_dense(&c, &format!("repetition d{d} r{r}"));
    }
}

#[test]
fn live_forward_matches_dense_when_every_qubit_stays_live() {
    let n = 130;
    let order: Vec<usize> = (0..n).map(|i| (i * 67) % n).collect();
    let c = measure_in_order(circuits::clifford_heavy_circuit(n, 2, 7), &order);
    assert_live_matches_dense(&c, "clifford 130 scrambled");

    let n = 200;
    let order: Vec<usize> = (0..n).map(|i| (i * 77) % n).collect();
    let c = measure_in_order(circuits::ghz_circuit(n), &order);
    assert_live_matches_dense(&c, "ghz 200 scrambled");

    let n = 150;
    let order: Vec<usize> = (0..n).rev().step_by(3).collect();
    let c = measure_in_order(circuits::clifford_heavy_circuit(n, 3, 11), &order);
    assert_live_matches_dense(&c, "clifford 150 partial register");
}

// Records on untouched qubits, a qubit read twice, a readout order that differs from
// the order the qubits fall idle, and the S/SX family on the way to the readout.
#[test]
fn live_forward_matches_dense_on_irregular_schedules() {
    let n = 96;
    let mut c = circuits::clifford_heavy_circuit(70, 3, 5);
    c.num_qubits = n;
    for q in 0..20 {
        c.add_gate(Gate::S, &[q]);
        c.add_gate(Gate::SX, &[q + 20]);
        c.add_gate(Gate::Sdg, &[q + 40]);
        c.add_gate(Gate::SXdg, &[q + 1]);
    }
    let mut order: Vec<usize> = (0..n).rev().collect();
    order.push(5);
    order.push(80);
    order.push(5);
    let c = measure_in_order(c, &order);
    assert_live_matches_dense(&c, "untouched tail and repeats");

    let mut c = Circuit::new(80, 0);
    for q in 0..40 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::Cx, &[q, 40 + q]);
    }
    for q in 0..39 {
        c.add_gate(Gate::Cz, &[40 + q, 41 + q]);
    }
    let order: Vec<usize> = (0..40).chain((40..80).rev()).collect();
    let c = measure_in_order(c, &order);
    assert_live_matches_dense(&c, "readout against idle order");
}

#[test]
fn live_forward_agrees_with_tableau_runs() {
    let c = defer_measure_reset_circuit(&surface_memory_mr(3, 12)).unwrap();
    let schedule = live_forward::LiveSchedule::plan(&c);
    assert!(schedule.peak_live * schedule.peak_live <= LIVE_FORWARD_DENSITY * c.num_qubits);
    assert_forward_matches_tableau(&c, 60);
}
