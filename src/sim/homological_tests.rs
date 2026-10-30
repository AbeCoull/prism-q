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
fn boundary_reset_returns_the_qubit_to_zero() {
    let mut circuit = crate::circuit::Circuit::new(1, 1);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    assert_eq!((ecc.boundary_dim(), ecc.homology_dim()), (1, 0));
}

#[test]
fn boundary_reset_of_a_bell_half_leaves_its_partner_mixed() {
    let mut circuit = crate::circuit::Circuit::new(2, 2);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
    circuit.add_reset(0);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    // Z on q1 commutes with the mixed state, so no Z-type error is logical.
    assert_eq!((ecc.boundary_dim(), ecc.homology_dim()), (2, 0));
}

#[test]
fn boundary_mid_circuit_measurement_collapses_its_qubit() {
    let mut circuit = crate::circuit::Circuit::new(1, 2);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_measure(0, 1);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    // The state ends in an X eigenstate, so a Z error flips it and no Z readout sees it.
    assert_eq!((ecc.boundary_dim(), ecc.homology_dim()), (0, 1));
}

#[test]
fn boundary_z_readout_after_a_mixed_reset_counts_again() {
    for reset_q1 in [true, false] {
        let mut circuit = crate::circuit::Circuit::new(2, 3);
        circuit.add_gate(crate::gates::Gate::H, &[0]);
        circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
        circuit.add_reset(0);
        if reset_q1 {
            circuit.add_reset(1);
        } else {
            circuit.add_measure(1, 2);
        }
        circuit.add_gate(crate::gates::Gate::H, &[1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
        // q1 ends in an X eigenstate, so a Z error on it is logical.
        assert_eq!(
            (ecc.boundary_dim(), ecc.homology_dim()),
            (1, 1),
            "reset_q1={reset_q1}"
        );
    }
}

#[test]
fn boundary_counts_a_circuit_with_no_final_measurement() {
    let mut circuit = crate::circuit::Circuit::new(1, 1);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    assert_eq!((ecc.boundary_dim(), ecc.homology_dim()), (1, 1));

    let mut unmeasured = crate::circuit::Circuit::new(2, 0);
    unmeasured.add_gate(crate::gates::Gate::H, &[0]);
    let noise = NoiseModel::uniform_depolarizing(&unmeasured, 0.001);
    let ecc = ErrorChainComplex::build(&unmeasured, &noise, 42).unwrap();
    assert_eq!((ecc.boundary_dim(), ecc.homology_dim()), (2, 2));
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
fn analytical_marginals_fold_in_readout_error() {
    // Deterministic bits under a readout-only model, so the closed form is the
    // whole answer: bit 0 is measured 1 and reads 1 with probability 1 - p10,
    // bit 1 is measured 0 and reads 1 with probability p01.
    let mut circuit = crate::circuit::Circuit::new(2, 2);
    circuit.add_gate(crate::gates::Gate::X, &[0]);
    circuit.measure_all();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.set_bit_readout_error(0, 0.02, 0.07);
    noise.set_bit_readout_error(1, 0.03, 0.11);

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    assert!((marginals[0] - 0.93).abs() < 1e-12, "{marginals:?}");
    assert!((marginals[1] - 0.03).abs() < 1e-12, "{marginals:?}");
}

#[test]
fn analytical_marginals_with_readout_match_sampled() {
    let n = 6;
    let mut circuit = circuits::ghz_circuit(n);
    circuit.measure_all();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    for bit in 0..n {
        noise.set_bit_readout_error(bit, 0.04, 0.09);
    }

    let analytical = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();

    let shots = 20_000;
    let sampled = crate::sim::noise::run_shots_noisy(&circuit, &noise, shots, 42).unwrap();
    let counts: Vec<f64> = (0..n)
        .map(|bit| sampled.shots.iter().filter(|record| record[bit]).count() as f64 / shots as f64)
        .collect();

    for bit in 0..n {
        // Five sigma on a binomial proportion at this shot count.
        let sigma = (analytical[bit] * (1.0 - analytical[bit]) / shots as f64).sqrt();
        assert!(
            (analytical[bit] - counts[bit]).abs() < 5.0 * sigma,
            "bit {bit}: analytical={:.6}, sampled={:.6}, sigma={sigma:.6}",
            analytical[bit],
            counts[bit],
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

// A terminal `measure_all` carries three of its four pre-measurement flips on
// the slots of the measurements before them.
#[test]
fn analytical_marginals_count_flips_on_measurement_slots() {
    let mut circuit = Circuit::new(4, 4);
    for q in 0..4 {
        circuit.add_gate(crate::gates::Gate::Id, &[q]);
    }
    circuit.measure_all();
    let noise = crate::sim::noise::NoiseBuilder::new()
        .before_measurements(crate::sim::noise::NoiseChannel::Pauli {
            px: 0.1,
            py: 0.0,
            pz: 0.0,
        })
        .build(&circuit)
        .unwrap();

    let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
    for (bit, p) in marginals.iter().enumerate() {
        assert!((p - 0.1).abs() < 1e-12, "bit {bit}: {p}");
    }
}

#[test]
fn a_noise_model_built_before_the_measurements_is_an_error() {
    let mut circuit = crate::circuit::Circuit::new(2, 2);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
    let noise = crate::sim::noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    circuit.measure_all();

    let err = noisy_marginals_analytical(&circuit, &noise, 42).unwrap_err();
    assert!(err.to_string().contains("event slots"), "{err}");
}

fn flip_then_conditional(
    flipped: usize,
    body: crate::gates::Gate,
) -> (crate::circuit::Circuit, NoiseModel) {
    let mut circuit = crate::circuit::Circuit::new(2, 2);
    circuit.add_gate(crate::gates::Gate::Id, &[flipped]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: crate::circuit::ClassicalCondition::BitIsOne(0),
        gate: body,
        targets: [1].as_slice().into(),
    });
    circuit.add_measure(1, 1);
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate.iter_mut().for_each(Vec::clear);
    noise.after_gate[0].push(crate::sim::noise::NoiseEvent::pauli(flipped, 0.1, 0.0, 0.0));
    (circuit, noise)
}

// Bit 0 always reads 0, so the guarded H never runs and the flip before it
// reaches bit 1 every time it fires.
#[test]
fn a_guarded_non_pauli_gate_is_rejected() {
    let (circuit, noise) = flip_then_conditional(1, crate::gates::Gate::H);
    let err = ErrorChainComplex::build(&circuit, &noise, 42).err();
    assert!(
        matches!(
            err,
            Some(crate::error::PrismError::IncompatibleBackend { .. })
        ),
        "{err:?}"
    );
}

// A flip of bit 0 also decides whether the guarded X flips bit 1, a path the
// backward propagation never sees.
#[test]
fn a_guarded_pauli_is_rejected() {
    let (circuit, noise) = flip_then_conditional(0, crate::gates::Gate::X);
    let err = ErrorChainComplex::build(&circuit, &noise, 42).err();
    assert!(
        matches!(
            err,
            Some(crate::error::PrismError::IncompatibleBackend { .. })
        ),
        "{err:?}"
    );
}

fn flipped_before(
    then: impl FnOnce(&mut crate::circuit::Circuit),
) -> (crate::circuit::Circuit, NoiseModel) {
    let mut circuit = crate::circuit::Circuit::new(1, 1);
    circuit.add_gate(crate::gates::Gate::Id, &[0]);
    then(&mut circuit);
    circuit.add_measure(0, 0);
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate.iter_mut().for_each(Vec::clear);
    noise.after_gate[0].push(crate::sim::noise::NoiseEvent::pauli(0, 0.1, 0.0, 0.0));
    (circuit, noise)
}

#[test]
fn a_reset_discards_an_earlier_flip() {
    let (circuit, noise) = flipped_before(|c| c.add_reset(0));
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    assert_eq!(ecc.noisy_marginals(&[0.0]), vec![0.0]);
}

#[test]
fn a_flip_after_a_reset_still_counts() {
    let (circuit, mut noise) = flipped_before(|c| c.add_reset(0));
    noise.after_gate.swap(0, 1);
    let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
    assert!((ecc.noisy_marginals(&[0.0])[0] - 0.1).abs() < 1e-12);
}

#[test]
fn a_non_clifford_gate_is_rejected() {
    let (circuit, noise) = flipped_before(|c| {
        c.add_gate(crate::gates::Gate::H, &[0]);
        c.add_gate(crate::gates::Gate::T, &[0]);
        c.add_gate(crate::gates::Gate::H, &[0]);
    });
    let err = ErrorChainComplex::build(&circuit, &noise, 42).err();
    assert!(
        matches!(
            err,
            Some(crate::error::PrismError::IncompatibleBackend { .. })
        ),
        "{err:?}"
    );
}

#[test]
fn a_non_pauli_channel_is_rejected_by_the_build() {
    let (circuit, _) = flipped_before(|_| {});
    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.1);
    let err = ErrorChainComplex::build(&circuit, &noise, 42).err();
    assert!(
        matches!(
            err,
            Some(crate::error::PrismError::IncompatibleBackend { .. })
        ),
        "{err:?}"
    );
}
