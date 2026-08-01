use super::*;
use crate::circuits;

#[test]
fn gf2_kernel_identity() {
    // Identity matrix: kernel is trivial (empty)
    let mut m = F2DenseMatrix::new(3, 3);
    m.set(0, 0);
    m.set(1, 1);
    m.set(2, 2);
    let k = gf2_kernel(&m);
    assert!(k.is_empty(), "Identity matrix should have trivial kernel");
}

#[test]
fn gf2_kernel_zero_matrix() {
    // Zero matrix: kernel is the full space
    let m = F2DenseMatrix::new(3, 4);
    let k = gf2_kernel(&m);
    assert_eq!(k.len(), 4, "Zero 3×4 matrix should have 4-dim kernel");
}

#[test]
fn gf2_kernel_rank_deficient() {
    // [1 1 0]
    // [0 1 1]
    // Row 0 + Row 1 = [1 0 1], so rank = 2, kernel dim = 3 - 2 = 1
    let mut m = F2DenseMatrix::new(2, 3);
    m.set(0, 0);
    m.set(0, 1);
    m.set(1, 1);
    m.set(1, 2);
    let k = gf2_kernel(&m);
    assert_eq!(k.len(), 1, "rank-2 2×3 matrix should have 1-dim kernel");
    // Kernel vector should be [1, 1, 1] (x₀ = x₁ = x₂)
    // Row 0: x₀ + x₁ = 0 → x₀ = x₁
    // Row 1: x₁ + x₂ = 0 → x₁ = x₂
    let kv = &k[0];
    assert_eq!(kv[0] & 0b111, 0b111, "kernel vector should be [1,1,1]");
}

#[test]
fn gf2_kernel_verifies() {
    // Verify Mx = 0 for all kernel vectors
    let mut m = F2DenseMatrix::new(3, 5);
    // Some arbitrary matrix
    m.set(0, 0);
    m.set(0, 2);
    m.set(0, 4);
    m.set(1, 1);
    m.set(1, 3);
    m.set(2, 0);
    m.set(2, 1);
    m.set(2, 2);

    let k = gf2_kernel(&m);
    for kv in &k {
        // Check M · kv = 0
        for r in 0..3 {
            let mut dot = 0u32;
            for c in 0..5 {
                if m.get(r, c) && (kv[c / 64] >> (c % 64)) & 1 != 0 {
                    dot ^= 1;
                }
            }
            assert_eq!(dot, 0, "kernel vector should satisfy Mx = 0");
        }
    }
}

#[test]
fn homological_ghz_compiles() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    assert!(sampler.syndrome_rank() <= n, "syndrome rank should be ≤ n");
}

#[test]
fn homological_ghz_samples() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let mut sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    let shots = sampler.sample_bulk(1000);
    assert_eq!(shots.len(), 1000);
    assert_eq!(shots[0].len(), n);
}

#[test]
fn homological_bell_pairs() {
    let n = 4;
    let mut circuit = circuits::independent_bell_pairs(n / 2);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    // Bell pairs with noise should have non-trivial syndrome rank
    assert!(sampler.syndrome_rank() > 0);
}

#[test]
fn homological_class_probs_sum_to_one() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    let sum: f64 = sampler.class_probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "class probabilities should sum to 1, got {sum}"
    );
}

#[test]
fn homological_matches_brute_force_statistics() {
    let n = 4;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.05);
    let num_shots = 10000;

    // Homological sampler
    let homo_result = run_shots_homological(&circuit, &noise, num_shots, 42).unwrap();

    // Brute-force sampler
    let brute_result = crate::sim::noise::run_shots_noisy(&circuit, &noise, num_shots, 42).unwrap();

    for bit in 0..n {
        let homo_p = homo_result.marginal(bit);
        let brute_p = brute_result.marginal(bit);
        let diff = (homo_p - brute_p).abs();
        assert!(
            diff < 0.05,
            "bit {bit}: homological p={homo_p:.4}, brute p={brute_p:.4}, diff={diff:.4}"
        );
    }
}

#[test]
fn boundary_trivial_circuit_has_zero_homology() {
    let n = 4;
    let mut circuit = crate::circuit::Circuit::new(n, n);
    for i in 0..n {
        circuit.add_measure(i, i);
    }
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    assert_eq!(ecc.boundary_dim(), n);
    assert_eq!(ecc.homology_dim(), 0);
}

#[test]
fn boundary_ghz_has_one_logical_qubit() {
    for n in [3, 5, 8] {
        let mut circuit = circuits::ghz_circuit(n);
        circuit.measure_all();
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
        assert_eq!(
            ecc.homology_dim(),
            1,
            "GHZ-{n} should have 1 logical error class"
        );
        assert_eq!(ecc.boundary_dim(), n - 1);
    }
}

#[test]
fn boundary_bell_pair_has_one_logical() {
    let mut circuit = crate::circuit::Circuit::new(2, 2);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    assert_eq!(ecc.homology_dim(), 1);
    assert_eq!(ecc.boundary_dim(), 1);
}

#[test]
fn boundary_independent_bell_pairs() {
    let n_pairs = 3;
    let mut circuit = circuits::independent_bell_pairs(n_pairs);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    assert_eq!(
        ecc.homology_dim(),
        n_pairs,
        "{n_pairs} bell pairs should have {n_pairs} logical error classes"
    );
}

#[test]
fn boundary_exposed_via_sampler() {
    let n = 4;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    assert_eq!(sampler.homology_dim(), 1);
    assert_eq!(sampler.boundary_dim(), n - 1);
}

#[test]
fn boundary_partial_measurement() {
    let mut circuit = crate::circuit::Circuit::new(3, 1);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    // 3 qubits, 1 measured: ker(σ) has dim 2*3-1=5
    // Stabilizers: X₀X₁, Z₀Z₁, Z₂ (3 generators)
    // X-projection on qubit 0: X₀X₁ has X on q0 → rank(A) = 1
    // boundary_dim = 3-1 = 2, homology_dim = 3-1+1 = 3
    assert_eq!(ecc.boundary_dim(), 2);
    assert_eq!(ecc.homology_dim(), 3);
}

#[test]
fn packed_matches_unpacked() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let mut s1 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    let mut s2 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();

    let unpacked = s1.sample_bulk(500);
    let packed = s2.sample_packed(500);

    assert_eq!(packed.num_shots(), 500);
    assert_eq!(packed.num_measurements(), n);

    for (s, shot) in unpacked.iter().enumerate() {
        for (m, &val) in shot.iter().enumerate() {
            assert_eq!(packed.get_bit(s, m), val, "mismatch at shot={s} meas={m}");
        }
    }
}

#[test]
fn marginals_matches_unpacked() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let mut s1 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    let mut s2 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();

    let num_shots = 10_000;
    let unpacked = s1.sample_bulk(num_shots);
    let marginals = s2.sample_marginals(num_shots);

    assert_eq!(marginals.len(), n);
    for m in 0..n {
        let unpacked_p = unpacked.iter().filter(|s| s[m]).count() as f64 / num_shots as f64;
        assert!(
            (marginals[m] - unpacked_p).abs() < 1e-10,
            "marginal mismatch at meas={m}: packed={}, unpacked={unpacked_p}",
            marginals[m],
        );
    }
}

#[test]
fn analytical_marginals_match_sampled_small() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let analytical = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();

    let mut sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    let sampled = sampler.sample_marginals(100_000);

    assert_eq!(analytical.len(), n);
    assert_eq!(sampled.len(), n);
    for i in 0..n {
        assert!(
            (analytical[i] - sampled[i]).abs() < 0.01,
            "bit {i}: analytical={:.6}, sampled={:.6}",
            analytical[i],
            sampled[i],
        );
    }
}

#[test]
fn analytical_marginals_ghz_50q() {
    let n = 50;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    assert_eq!(marginals.len(), n);
    for (i, &p) in marginals.iter().enumerate() {
        assert!(p > 0.0 && p < 1.0, "bit {i}: marginal {p} out of range");
        assert!(
            (p - 0.5).abs() < 0.05,
            "bit {i}: GHZ marginal should be near 0.5, got {p}"
        );
    }
}

#[test]
fn analytical_marginals_ghz_100q() {
    let n = 100;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    assert_eq!(marginals.len(), n);
    for (i, &p) in marginals.iter().enumerate() {
        assert!(p > 0.0 && p < 1.0, "bit {i}: marginal {p} out of range");
    }
}

#[test]
fn analytical_marginals_bell_pairs_100q() {
    let n_pairs = 50;
    let n = n_pairs * 2;
    let mut circuit = circuits::independent_bell_pairs(n_pairs);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    assert_eq!(marginals.len(), n);
    for (i, &p) in marginals.iter().enumerate() {
        assert!(
            (p - 0.5).abs() < 0.05,
            "bit {i}: bell pair marginal should be near 0.5, got {p}"
        );
    }
}

#[test]
fn analytical_marginals_clifford_1000q() {
    let n = 1000;
    let mut circuit = circuits::clifford_heavy_circuit(n, 2, 42);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    assert_eq!(marginals.len(), n);
    for (i, &p) in marginals.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "bit {i}: marginal {p} out of range"
        );
    }
}

#[test]
fn analytical_marginals_deterministic_bits() {
    let mut circuit = crate::circuit::Circuit::new(4, 4);
    for i in 0..4 {
        circuit.add_gate(crate::gates::Gate::X, &[i]);
    }
    for i in 0..4 {
        circuit.add_measure(i, i);
    }
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    for (i, &p) in marginals.iter().enumerate() {
        assert!(
            p > 0.95,
            "bit {i}: X-then-measure should give p(1) near 1.0, got {p}"
        );
    }
}

#[test]
fn analytical_marginals_no_noise() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    for (i, &p) in marginals.iter().enumerate() {
        assert!(
            (p - 0.5).abs() < 1e-10,
            "bit {i}: GHZ with no noise should have marginal 0.5, got {p}"
        );
    }
}

#[test]
fn chunked_accumulator_matches_packed() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

    let mut s1 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
    let mut s2 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();

    let num_shots = 5_000;
    let packed = s1.sample_packed(num_shots);
    let direct_counts = packed.counts();

    let mut acc = super::super::compiled::HistogramAccumulator::new();
    s2.sample_chunked(num_shots, &mut acc);
    let chunked_counts = acc.into_counts();

    assert_eq!(direct_counts, chunked_counts);
}
