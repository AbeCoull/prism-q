use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::{FRAC_PI_4, SQRT_2};

fn t_branch() -> RotBranch {
    RotBranch::new(FRAC_PI_4)
}

#[test]
fn test_no_t_gates_matches_propagate_backward() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::S, &[2]);

    let obs = PauliVec::z_on_qubit(1, 1);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let ops = coalesce_cliffords(&circuit);
    let (result, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);

    let mut expected = PauliVec::z_on_qubit(1, 1);
    for inst in circuit.instructions.iter().rev() {
        if let Instruction::Gate { gate, targets } = inst {
            propagate_backward(&mut expected, gate, targets);
        }
    }

    assert_eq!(result.x, expected.x);
    assert_eq!(result.z, expected.z);
    assert!((weight - Complex64::new(1.0, 0.0)).norm() < 1e-14);
}

#[test]
fn test_h_t_h_expectation_converges() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);

    let obs = PauliVec::z_on_qubit(1, 0);
    let num_samples = 100_000;
    let mut sum = 0.0;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let ops = coalesce_cliffords(&circuit);

    for _ in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);
        if pauli.is_diagonal() {
            sum += weight.re;
        }
    }
    let mean = sum / num_samples as f64;

    let exact_z = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (mean - exact_z).abs() < 0.02,
        "mean={mean}, expected≈{exact_z}"
    );
}

#[test]
fn test_branch_t_gate_passthrough_iz() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut pauli_i = PauliVec::new(1);
    let w = branch_z_rotation(&mut pauli_i, 0, &t_branch(), &mut rng);
    assert!((w - Complex64::new(1.0, 0.0)).norm() < 1e-14);

    let mut pauli_z = PauliVec::z_on_qubit(1, 0);
    let w = branch_z_rotation(&mut pauli_z, 0, &t_branch(), &mut rng);
    assert!((w - Complex64::new(1.0, 0.0)).norm() < 1e-14);
}

#[test]
fn test_branch_t_gate_x_branches() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let num_samples = 10_000;
    let mut x_count = 0;
    let mut y_count = 0;

    for _ in 0..num_samples {
        let mut pauli = PauliVec::new(1);
        pauli.x[0] = 1;
        let w = branch_z_rotation(&mut pauli, 0, &t_branch(), &mut rng);
        assert!((w.norm() - SQRT_2).abs() < 1e-14);
        if pauli.z[0] == 0 {
            x_count += 1;
        } else {
            y_count += 1;
        }
    }

    let ratio = x_count as f64 / num_samples as f64;
    assert!(
        (ratio - 0.5).abs() < 0.03,
        "expected ~50/50, got {x_count}/{y_count}"
    );
}

fn marginal_p0(full_probs: &[f64], _n: usize, qubit: usize) -> f64 {
    let mut p0 = 0.0;
    for (i, &p) in full_probs.iter().enumerate() {
        if (i >> qubit) & 1 == 0 {
            p0 += p;
        }
    }
    p0
}

#[test]
fn test_run_spp_vs_statevector_3q_2t() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::H, &[0]);

    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();

    let spp = run_spp(&circuit, 200_000, 42).unwrap();
    assert_eq!(spp.t_count, 2);

    for q in 0..3 {
        let exact_p0 = marginal_p0(&sv_probs, 3, q);
        let exact_ez = 2.0 * exact_p0 - 1.0;
        let err = (spp.expectations[q] - exact_ez).abs();
        assert!(
            err < 3.0 * spp.std_errors[q] + 0.01,
            "qubit {q}: spp={}, exact={exact_ez}, err={err}, 3σ={}",
            spp.expectations[q],
            3.0 * spp.std_errors[q]
        );
    }
}

#[test]
fn test_run_spp_vs_statevector_4q_4t() {
    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_gate(Gate::T, &[2]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::T, &[3]);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }

    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();

    let spp = run_spp(&circuit, 200_000, 42).unwrap();
    assert_eq!(spp.t_count, 4);

    for q in 0..4 {
        let exact_p0 = marginal_p0(&sv_probs, 4, q);
        let exact_ez = 2.0 * exact_p0 - 1.0;
        let err = (spp.expectations[q] - exact_ez).abs();
        assert!(
            err < 3.0 * spp.std_errors[q] + 0.01,
            "qubit {q}: spp={}, exact={exact_ez}, err={err}, 3σ={}",
            spp.expectations[q],
            3.0 * spp.std_errors[q]
        );
    }
}

#[test]
fn test_spp_to_probabilities() {
    let result = SppResult {
        expectations: vec![0.5, -0.3],
        std_errors: vec![0.01, 0.01],
        num_samples: 1000,
        t_count: 2,
        nonzero_fraction: 0.8,
    };
    let probs = spp_to_probabilities(&result);
    assert_eq!(probs.len(), 4);
    assert!((probs[0] - 0.75).abs() < 1e-14);
    assert!((probs[1] - 0.25).abs() < 1e-14);
    assert!((probs[2] - 0.35).abs() < 1e-14);
    assert!((probs[3] - 0.65).abs() < 1e-14);
}

#[test]
fn test_spd_no_t_gates_exact() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::S, &[2]);

    let spd = run_spd(&circuit, 0.0, 0).unwrap();
    assert_eq!(spd.t_count, 0);
    assert_eq!(spd.max_terms, 1);

    let spp = run_spp(&circuit, 10_000, 42).unwrap();
    for q in 0..3 {
        assert!(
            (spd.expectations[q] - spp.expectations[q]).abs() < 0.05,
            "qubit {q}: spd={}, spp={}",
            spd.expectations[q],
            spp.expectations[q]
        );
    }
}

#[test]
fn test_spd_h_t_h_exact() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);

    let spd = run_spd(&circuit, 0.0, 0).unwrap();
    let exact_z = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (spd.expectations[0] - exact_z).abs() < 1e-10,
        "spd={}, exact={exact_z}",
        spd.expectations[0]
    );
}

#[test]
fn test_spd_vs_statevector_3q_2t() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::H, &[0]);

    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    let spd = run_spd(&circuit, 0.0, 0).unwrap();
    assert_eq!(spd.t_count, 2);

    for q in 0..3 {
        let exact_p0 = marginal_p0(&sv_probs, 3, q);
        let exact_ez = 2.0 * exact_p0 - 1.0;
        assert!(
            (spd.expectations[q] - exact_ez).abs() < 1e-10,
            "qubit {q}: spd={}, exact={exact_ez}",
            spd.expectations[q]
        );
    }
}

#[test]
fn test_spd_vs_statevector_4q_4t() {
    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_gate(Gate::T, &[2]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::T, &[3]);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }

    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    let spd = run_spd(&circuit, 0.0, 0).unwrap();
    assert_eq!(spd.t_count, 4);

    for q in 0..4 {
        let exact_p0 = marginal_p0(&sv_probs, 4, q);
        let exact_ez = 2.0 * exact_p0 - 1.0;
        assert!(
            (spd.expectations[q] - exact_ez).abs() < 1e-10,
            "qubit {q}: spd={}, exact={exact_ez}",
            spd.expectations[q]
        );
    }
}

#[test]
fn test_spd_truncation() {
    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::T, &[2]);
    circuit.add_gate(Gate::T, &[3]);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }

    let exact = run_spd(&circuit, 0.0, 0).unwrap();
    let approx = run_spd(&circuit, 1e-6, 0).unwrap();

    for q in 0..4 {
        assert!(
            (exact.expectations[q] - approx.expectations[q]).abs() < 1e-4,
            "qubit {q}: exact={}, approx={}",
            exact.expectations[q],
            approx.expectations[q]
        );
    }
}

#[test]
fn test_spd_with_tdg() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Tdg, &[0]);
    circuit.add_gate(Gate::H, &[0]);

    let spd = run_spd(&circuit, 0.0, 0).unwrap();
    let exact_z = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (spd.expectations[0] - exact_z).abs() < 1e-10,
        "spd={}, exact={exact_z}",
        spd.expectations[0]
    );
}

#[test]
fn test_spd_to_probabilities() {
    let result = SpdResult {
        expectations: vec![0.5, -0.3],
        t_count: 2,
        max_terms: 4,
        total_discarded: 0.0,
    };
    let probs = spd_to_probabilities(&result);
    assert_eq!(probs.len(), 4);
    assert!((probs[0] - 0.75).abs() < 1e-14);
    assert!((probs[1] - 0.25).abs() < 1e-14);
    assert!((probs[2] - 0.35).abs() < 1e-14);
    assert!((probs[3] - 0.65).abs() < 1e-14);
}

#[test]
fn test_spd_h_t_h_t_h_phase_regression() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);

    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    let exact_ez = 2.0 * sv_probs[0] - 1.0;

    let spd = run_spd(&circuit, 0.0, 0).unwrap();
    assert!(
        (spd.expectations[0] - exact_ez).abs() < 1e-10,
        "spd={}, exact={exact_ez}",
        spd.expectations[0]
    );
}

#[test]
fn test_spd_vs_statevector_14q_clifford_t() {
    let c = crate::circuits::clifford_t_circuit(14, 10, 0.1, 42);
    let spd = run_spd(&c, 0.0, 0).unwrap();

    let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();

    for q in 0..14 {
        let exact_p0 = marginal_p0(&sv_probs, 14, q);
        let exact_ez = 2.0 * exact_p0 - 1.0;
        assert!(
            (spd.expectations[q] - exact_ez).abs() < 1e-8,
            "qubit {q}: spd={}, exact={exact_ez}",
            spd.expectations[q]
        );
    }
}

#[test]
fn coalesce_cliffords_long_clifford_run_stays_phase_correct() {
    // Regression: the SmallCliff path threads
    // `clifford_conjugation_phase` through every gate so global
    // signs from `HYH = -Y`, `SYS† = -X`, etc. are preserved.
    // Confirm SPP matches the analytical SPD on a long Clifford
    // run.
    let mut circuit = Circuit::new(4, 0);
    for _ in 0..20 {
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
            circuit.add_gate(Gate::S, &[q]);
        }
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::Cz, &[2, 3]);
    }
    let spp = run_spp(&circuit, 4_000, 42).unwrap();
    let spd = run_spd(&circuit, 1e-10, 16_384).unwrap();
    for q in 0..4 {
        assert!(
            (spp.expectations[q] - spd.expectations[q]).abs() < 0.08,
            "qubit {q}: spp={}, spd={}",
            spp.expectations[q],
            spd.expectations[q]
        );
    }
}

#[test]
fn run_spp_pure_clifford() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let result = run_spp(&circuit, 16, 42).unwrap();
    assert_eq!(result.expectations.len(), 2);
    assert!(result.t_count == 0);
}

#[test]
fn run_spd_pure_clifford() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let result = run_spd(&circuit, 0.0, 1024).unwrap();
    let probs = spd_to_probabilities(&result);
    assert_eq!(probs.len(), 4);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 2.0).abs() < 1e-9);
}

#[test]
fn light_cone_excludes_disjoint_gates() {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[3]);
    circuit.add_gate(Gate::T, &[3]);
    circuit.add_gate(Gate::H, &[3]);

    let obs = [PauliTerm::z(0)];
    let keep = inverse_light_cone(&circuit, &obs);
    assert_eq!(keep, vec![true, true, true, false, false, false]);
}

#[test]
fn light_cone_follows_entangling_gates() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::H, &[0]);

    let obs = [PauliTerm::z(2)];
    let keep = inverse_light_cone(&circuit, &obs);
    assert_eq!(keep, vec![true, true, true, false]);
}

#[test]
fn light_cone_spd_matches_unrestricted_spd() {
    let mut circuit = Circuit::new(5, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[4]);
    circuit.add_gate(Gate::T, &[4]);
    circuit.add_gate(Gate::Cx, &[3, 4]);
    circuit.add_gate(Gate::T, &[3]);

    let obs = [PauliTerm::z(0)];
    let full = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
    let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
    assert!(
        (full.mean - cone.mean).abs() < 1e-12,
        "full={} cone={}",
        full.mean,
        cone.mean
    );
    assert!(cone.peak_terms <= full.peak_terms);
}

#[test]
fn light_cone_skips_most_gates_on_disjoint_t_block() {
    let mut circuit = Circuit::new(6, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);
    for _ in 0..6 {
        circuit.add_gate(Gate::H, &[3]);
        circuit.add_gate(Gate::T, &[3]);
        circuit.add_gate(Gate::Cx, &[3, 4]);
        circuit.add_gate(Gate::T, &[4]);
        circuit.add_gate(Gate::Cx, &[4, 5]);
        circuit.add_gate(Gate::T, &[5]);
    }

    let obs = [PauliTerm::z(0)];
    let keep = inverse_light_cone(&circuit, &obs);
    let kept = keep.iter().filter(|b| **b).count();
    assert_eq!(kept, 3, "only the H-T-H block on q0 should be kept");

    let full = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
    let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
    assert!((full.mean - cone.mean).abs() < 1e-12);
}

#[test]
fn light_cone_spd_matches_on_entangled_observable() {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::T, &[2]);
    circuit.add_gate(Gate::H, &[3]);
    circuit.add_gate(Gate::T, &[3]);

    let obs = [PauliTerm::z(0), PauliTerm::z(2)];
    let full = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
    let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
    assert!(
        (full.mean - cone.mean).abs() < 1e-12,
        "full={} cone={}",
        full.mean,
        cone.mean
    );
}
