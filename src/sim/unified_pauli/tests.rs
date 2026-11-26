use super::spd::*;
use super::spp::*;
use super::*;
use crate::limits::ResolvedLimits;
use crate::sim::compiled::propagate_backward;
use crate::sim::noise::NoiseModel;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::{FRAC_PI_4, SQRT_2};

fn marginals_flat(expectations: &[f64]) -> Vec<f64> {
    crate::sim::expectations_to_marginals(expectations)
        .into_iter()
        .flat_map(|(p0, p1)| [p0, p1])
        .collect()
}

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
    let w = branch_z_rotation(&mut pauli_i, 0, ZROT_SINGLE, &t_branch(), &mut rng);
    assert!((w - Complex64::new(1.0, 0.0)).norm() < 1e-14);

    let mut pauli_z = PauliVec::z_on_qubit(1, 0);
    let w = branch_z_rotation(&mut pauli_z, 0, ZROT_SINGLE, &t_branch(), &mut rng);
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
        let w = branch_z_rotation(&mut pauli, 0, ZROT_SINGLE, &t_branch(), &mut rng);
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

    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &circuit,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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

    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &circuit,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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
    let probs = marginals_flat(&result.expectations);
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

    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &circuit,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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

    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &circuit,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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
    let probs = marginals_flat(&result.expectations);
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

    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &circuit,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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

    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        &c,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
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
    let probs = marginals_flat(&result.expectations);
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

// The ops slice streams through the SPP per-sample loop; a wider enum taxes
// every circuit (measured +2.6% to +5.8% on op-heavy rows at 48 bytes).
#[test]
fn coalesced_op_stays_at_the_one_qubit_variant_size() {
    assert_eq!(std::mem::size_of::<CoalescedOp>(), 40);
}

fn u3(theta: f64, phi: f64, lam: f64) -> Gate {
    Gate::Fused(Box::new(crate::circuit::gate_library::u_matrix(
        theta, phi, lam,
    )))
}

fn assert_spd_matches_statevector(circuit: &Circuit, tol: f64) {
    let sv = crate::sim::run_with(
        crate::sim::BackendKind::Statevector,
        circuit,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
    let sv_probs = sv.probabilities.unwrap().to_vec();
    let spd = run_spd(circuit, 0.0, 0).unwrap();
    for q in 0..circuit.num_qubits {
        let exact_ez = 2.0 * marginal_p0(&sv_probs, circuit.num_qubits, q) - 1.0;
        assert!(
            (spd.expectations[q] - exact_ez).abs() < tol,
            "qubit {q}: spd={}, exact={exact_ez}",
            spd.expectations[q]
        );
    }
}

// The parser spells `u3` as a `Fused` matrix and `rzz` is not a stdgate, so a
// transpiled QAOA arrives as `u3` plus `cx` with `Rz` as `u3(0, 0, gamma)`.
#[test]
fn spd_takes_the_u3_cx_spelling_of_qaoa() {
    let named = crate::circuits::qaoa_circuit(6, 2, 0xDEAD_BEEF);
    let mut qasm = String::from("OPENQASM 3.0;\nqubit[6] q;\n");
    for inst in &named.instructions {
        let Instruction::Gate { gate, targets } = inst else {
            unreachable!()
        };
        match gate {
            Gate::Rzz(gamma) => {
                let (a, b) = (targets[0], targets[1]);
                qasm.push_str(&format!(
                    "cx q[{a}], q[{b}];\nu3(0, 0, {gamma:.17}) q[{b}];\ncx q[{a}], q[{b}];\n"
                ));
            }
            Gate::Rx(beta) => {
                qasm.push_str(&format!("u3({beta:.17}, -pi/2, pi/2) q[{}];\n", targets[0]))
            }
            other => unreachable!("{other:?}"),
        }
    }
    let respelled = crate::circuit::openqasm::parse(&qasm).unwrap();
    assert!(respelled.instructions.iter().any(|inst| matches!(
        inst,
        Instruction::Gate {
            gate: Gate::Fused(_),
            ..
        }
    )));

    let reference = run_spd(&named, 0.0, 0).unwrap();
    let lowered = run_spd(&respelled, 0.0, 0).unwrap();
    for (q, (a, b)) in reference
        .expectations
        .iter()
        .zip(&lowered.expectations)
        .enumerate()
    {
        assert!((a - b).abs() < 1e-9, "qubit {q}: named={a}, u3={b}");
    }
}

#[test]
fn spd_lowers_general_matrices_and_controlled_targets() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(u3(0.3, 0.7, -1.1), &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(u3(2.2, 0.0, 0.4), &[1]);
    circuit.add_gate(Gate::cu(Gate::Y.matrix_2x2()), &[1, 2]);
    circuit.add_gate(Gate::cphase(0.9), &[0, 2]);
    circuit.add_gate(u3(std::f64::consts::PI, 0.5, 0.25), &[2]);
    circuit.add_gate(Gate::cu(Gate::Rz(1.3).matrix_2x2()), &[2, 0]);
    assert_spd_matches_statevector(&circuit, 1e-10);
}

#[test]
fn a_rotation_merges_each_flip_into_a_term_already_present() {
    let (a, b) = (0.37, 1.1);
    let mut split = Circuit::new(1, 0);
    split.add_gate(Gate::H, &[0]);
    split.add_gate(Gate::Rz(a), &[0]);
    split.add_gate(Gate::Rz(b), &[0]);
    let mut joined = Circuit::new(1, 0);
    joined.add_gate(Gate::H, &[0]);
    joined.add_gate(Gate::Rz(a + b), &[0]);

    let obs = [PauliTerm::x(0)];
    let split = run_spd_observable(&split, &obs, 0.0, 0).unwrap();
    let joined = run_spd_observable(&joined, &obs, 0.0, 0).unwrap();
    assert_eq!(split.peak_terms, 2);
    assert!((split.mean - joined.mean).abs() < 1e-12);
    assert!((split.mean - (a + b).cos()).abs() < 1e-12);
}

fn near_clifford_rotations(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::Rz(0.3 + 0.1 * q as f64), &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
        circuit.add_gate(Gate::T, &[q + 1]);
        circuit.add_gate(Gate::Rzz(0.7), &[q, q + 1]);
    }
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit
}

#[test]
fn one_pass_over_a_weighted_sum_matches_the_per_term_values() {
    let circuit = near_clifford_rotations(6);
    let terms: Vec<(f64, Vec<PauliTerm>)> = vec![
        (0.5, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (-1.25, vec![PauliTerm::x(2)]),
        (2.0, vec![PauliTerm::y(3), PauliTerm::z(5)]),
        (0.75, vec![PauliTerm::z(0), PauliTerm::z(1)]),
    ];
    let exact = SpdTruncation::Threshold {
        epsilon: 0.0,
        max_terms: 0,
    };
    let per_term: f64 = terms
        .iter()
        .map(|(w, obs)| w * run_spd_observable_with(&circuit, obs, &exact).unwrap().mean)
        .sum();
    let merged = run_spd_weighted_observable_with(
        &circuit,
        &terms
            .iter()
            .map(|(w, obs)| (*w, obs.as_slice()))
            .collect::<Vec<_>>(),
        &exact,
    )
    .unwrap();
    assert!((merged.mean - per_term).abs() < 1e-12);
    assert_eq!(merged.total_discarded, 0.0);
}

#[test]
fn a_term_budget_caps_the_merged_sum_as_a_whole() {
    let circuit = near_clifford_rotations(8);
    let terms: Vec<(f64, Vec<PauliTerm>)> = (0..8).map(|q| (1.0, vec![PauliTerm::x(q)])).collect();
    let budget = SpdTruncation::Budget { max_terms: 16 };
    let merged = run_spd_weighted_observable_with(
        &circuit,
        &terms
            .iter()
            .map(|(w, obs)| (*w, obs.as_slice()))
            .collect::<Vec<_>>(),
        &budget,
    )
    .unwrap();
    assert!(merged.peak_terms <= 16);
    assert!(merged.total_discarded > 0.0);
    let exact = run_spd_weighted_observable_with(
        &circuit,
        &terms
            .iter()
            .map(|(w, obs)| (*w, obs.as_slice()))
            .collect::<Vec<_>>(),
        &SpdTruncation::Threshold {
            epsilon: 0.0,
            max_terms: 0,
        },
    )
    .unwrap();
    assert!((merged.mean - exact.mean).abs() <= merged.total_discarded + 1e-12);
}

#[test]
fn a_merged_sum_past_the_term_ceiling_falls_back_to_one_pass_per_term() {
    let n = 21;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::Rz(0.3), &[q]);
    }
    let wide: Vec<PauliTerm> = (0..20).map(PauliTerm::x).collect();
    let narrow = vec![PauliTerm::x(20)];
    let exact = SpdTruncation::Threshold {
        epsilon: 0.0,
        max_terms: 0,
    };
    let wide_alone = run_spd_observable_with(&circuit, &wide, &exact).unwrap();
    let narrow_alone = run_spd_observable_with(&circuit, &narrow, &exact).unwrap();
    assert_eq!(wide_alone.peak_terms, SPD_MAX_TERMS_CEILING);

    let merged = run_spd_weighted_observable_with(
        &circuit,
        &[(0.5, wide.as_slice()), (2.0, narrow.as_slice())],
        &exact,
    )
    .unwrap();
    let expected = 0.5 * wide_alone.mean + 2.0 * narrow_alone.mean;
    assert!((merged.mean - expected).abs() < 1e-12);
    assert_eq!(merged.peak_terms, SPD_MAX_TERMS_CEILING);
}

fn rotation_mesh_on(num_qubits: usize, qubits: &[usize; 6]) -> Circuit {
    let mut c = Circuit::new(num_qubits, 0);
    for (layer, angle) in [0.31f64, -0.47, 0.73].into_iter().enumerate() {
        for (k, &q) in qubits.iter().enumerate() {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::Ry(angle + 0.05 * k as f64), &[q]);
            c.add_gate(Gate::T, &[q]);
            c.add_gate(Gate::Rz(angle * 1.3), &[q]);
            if (layer + k) % 2 == 0 {
                c.add_gate(Gate::SX, &[q]);
            }
        }
        for k in 0..5 {
            c.add_gate(Gate::Cx, &[qubits[k], qubits[k + 1]]);
            c.add_gate(
                Gate::Rzz(0.21 * (k + 1) as f64),
                &[qubits[k + 1], qubits[(k + 3) % 6]],
            );
        }
        c.add_gate(Gate::Cz, &[qubits[5], qubits[0]]);
        c.add_gate(Gate::S, &[qubits[layer]]);
    }
    c
}

#[test]
fn spd_engines_agree_across_key_widths() {
    let layouts: [(usize, [usize; 6]); 3] = [
        (6, [0, 1, 2, 3, 4, 5]),
        (128, [62, 63, 64, 65, 126, 127]),
        (200, [63, 64, 127, 128, 129, 199]),
    ];
    let mut reference: Option<[f64; 5]> = None;
    for (n, qubits) in layouts {
        let circuit = rotation_mesh_on(n, &qubits);
        let obs = [
            PauliTerm::x(qubits[2]),
            PauliTerm::z(qubits[4]),
            PauliTerm::y(qubits[5]),
        ];
        let exact = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap().mean;
        let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0)
            .unwrap()
            .mean;
        let noise = NoiseModel::with_amplitude_damping(&circuit, 0.03);
        let path = run_pauli_path_observable(&circuit, &noise, &obs, 0.0, 0)
            .unwrap()
            .mean;
        let marginals = run_spd(&circuit, 0.0, 0).unwrap().expectations;
        let values = [
            exact,
            cone,
            path,
            marginals[qubits[1]],
            marginals[qubits[3]],
        ];
        assert!((exact - cone).abs() < 1e-12, "{n} qubits");
        match reference {
            None => reference = Some(values),
            Some(expected) => {
                for (got, want) in values.iter().zip(expected) {
                    assert!((got - want).abs() < 1e-12, "{n} qubits: {got} vs {want}");
                }
            }
        }
    }
    let values = reference.unwrap();
    assert!(values[..3].iter().all(|v| v.abs() > 1e-6));
}

#[test]
fn threshold_truncation_drops_below_epsilon_and_reports_the_one_norm() {
    let coeffs = [
        Complex64::new(0.6, -0.2),
        Complex64::new(3e-4, 4e-4),
        Complex64::new(0.0, -7e-4),
        Complex64::new(1e-3, 0.0),
        Complex64::new(-2e-5, 1e-5),
    ];
    let build = || {
        let mut sum = WeightedPauliSum::<InlinePauli<1>>::new();
        for (k, &c) in coeffs.iter().enumerate() {
            sum.insert(InlinePauli::<1>::from_words(&[k as u64], &[0]), c);
        }
        sum
    };
    let epsilon = 1e-3;
    let expected: f64 = coeffs
        .iter()
        .map(|c| c.norm())
        .filter(|&m| m < epsilon)
        .sum();
    let mut sum = build();
    assert!((sum.truncate(epsilon) - expected).abs() < 1e-15);
    assert_eq!(sum.terms.len(), 2);

    let mut sum = build();
    assert_eq!(sum.truncate(-1.0), 0.0);
    assert_eq!(sum.terms.len(), coeffs.len());
}

fn reference_expectations(
    circuit: &Circuit,
    kind: crate::sim::BackendKind,
    noise: Option<&NoiseModel>,
    observables: &[Vec<PauliTerm>],
) -> Vec<f64> {
    let mut run = crate::sim::simulate(circuit).backend(kind);
    if let Some(noise) = noise {
        run = run.noise(noise);
    }
    run.seed(42).expectation_values(observables).unwrap()
}

fn native_axis_mix() -> Circuit {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.41), &[0]);
    c.add_gate(Gate::Ry(-0.83), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.29), &[1]);
    c.add_gate(Gate::Rx(0.2), &[1]);
    c.add_gate(Gate::Rx(-0.5), &[1]);
    c.add_gate(Gate::Ry(1.17), &[0]);
    c.add_gate(Gate::Rx(0.66), &[0]);
    c.add_gate(Gate::Rzz(0.66), &[1, 2]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::Rx(-1.9), &[2]);
    c.add_gate(Gate::Ry(0.37), &[2]);
    c.add_gate(Gate::Ry(0.52), &[2]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::SX, &[3]);
    c.add_gate(Gate::Ry(2.4), &[3]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Rx(0.73), &[3]);
    c.add_gate(Gate::Cx, &[3, 0]);
    c.add_gate(Gate::Ry(-0.61), &[0]);
    c.add_gate(Gate::Rzz(-0.4), &[0, 2]);
    c.add_gate(Gate::Rx(1.05), &[2]);
    c
}

fn native_axis_observables() -> Vec<Vec<PauliTerm>> {
    let mut observables: Vec<Vec<PauliTerm>> = (0..4)
        .flat_map(|q| [PauliTerm::x(q), PauliTerm::y(q), PauliTerm::z(q)])
        .map(|term| vec![term])
        .collect();
    observables.push(vec![PauliTerm::x(0), PauliTerm::y(2)]);
    observables.push(vec![PauliTerm::z(1), PauliTerm::x(2), PauliTerm::y(3)]);
    observables.push(vec![PauliTerm::y(0), PauliTerm::z(3)]);
    observables
}

#[test]
fn native_axis_rotations_match_the_statevector_on_every_letter() {
    let circuit = native_axis_mix();
    let observables = native_axis_observables();
    let want = reference_expectations(
        &circuit,
        crate::sim::BackendKind::Statevector,
        None,
        &observables,
    );
    let noiseless = NoiseModel::with_amplitude_damping(&circuit, 0.0);
    for (obs, want) in observables.iter().zip(&want) {
        let exact = run_spd_observable(&circuit, obs, 0.0, 0).unwrap().mean;
        let cone = run_spd_observable_light_cone(&circuit, obs, 0.0, 0)
            .unwrap()
            .mean;
        let path = run_pauli_path_observable(&circuit, &noiseless, obs, 0.0, 0)
            .unwrap()
            .mean;
        for (label, got) in [("spd", exact), ("light cone", cone), ("path", path)] {
            assert!(
                (got - want).abs() < 1e-12,
                "{label} {obs:?}: {got} vs {want}"
            );
        }
    }
    assert!(want.iter().filter(|v| v.abs() > 1e-3).count() > 6);

    let weights = [0.5, -1.25, 2.0, 0.75];
    let weighted: Vec<(f64, &[PauliTerm])> = observables[12..]
        .iter()
        .chain(&observables[1..2])
        .zip(weights)
        .map(|(obs, w)| (w, obs.as_slice()))
        .collect();
    let joint = run_spd_weighted_observable_with(
        &circuit,
        &weighted,
        &SpdTruncation::Threshold {
            epsilon: 0.0,
            max_terms: 0,
        },
    )
    .unwrap();
    let joint_want: f64 = weights[..3]
        .iter()
        .zip(&want[12..])
        .map(|(w, v)| w * v)
        .sum::<f64>()
        + weights[3] * want[1];
    assert!((joint.mean - joint_want).abs() < 1e-12);

    let marginals = run_spd(&circuit, 0.0, 0).unwrap().expectations;
    for q in 0..4 {
        let want = want[3 * q + 2];
        assert!(
            (marginals[q] - want).abs() < 1e-12,
            "qubit {q}: {} vs {want}",
            marginals[q]
        );
    }
}

#[test]
fn consecutive_axis_rotations_merge_each_flip_into_a_term_already_present() {
    let (a, b) = (0.37, 1.1);
    let rotations: [fn(f64) -> Gate; 2] = [Gate::Rx, Gate::Ry];
    let anticommuting = [
        [PauliTerm::z(0), PauliTerm::y(0)],
        [PauliTerm::z(0), PauliTerm::x(0)],
    ];
    for (rotation, letters) in rotations.into_iter().zip(anticommuting) {
        let mut split = Circuit::new(1, 0);
        split.add_gate(rotation(a), &[0]);
        split.add_gate(rotation(b), &[0]);
        let mut joined = Circuit::new(1, 0);
        joined.add_gate(rotation(a + b), &[0]);
        for letter in letters {
            let obs = [letter];
            let split = run_spd_observable(&split, &obs, 0.0, 0).unwrap();
            let joined = run_spd_observable(&joined, &obs, 0.0, 0).unwrap();
            assert_eq!(split.peak_terms, 2);
            assert!((split.mean - joined.mean).abs() < 1e-12);
        }
        let z = run_spd_observable(&split, &[PauliTerm::z(0)], 0.0, 0).unwrap();
        assert!((z.mean - (a + b).cos()).abs() < 1e-12);
    }
}

#[test]
fn native_axis_rotations_match_the_density_matrix_under_noise() {
    let circuit = native_axis_mix();
    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.04);
    let observables = native_axis_observables();
    let want = reference_expectations(
        &circuit,
        crate::sim::BackendKind::DensityMatrix,
        Some(&noise),
        &observables,
    );
    for (obs, want) in observables.iter().zip(&want) {
        let got = run_pauli_path_observable(&circuit, &noise, obs, 0.0, 0)
            .unwrap()
            .mean;
        assert!((got - want).abs() < 1e-12, "{obs:?}: {got} vs {want}");
    }
}

#[test]
fn branching_gate_count_is_unchanged_by_native_axis_rotations() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Rx(0.3), &[0]);
    circuit.add_gate(Gate::Ry(-0.7), &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::Rzz(0.2), &[0, 1]);
    circuit.add_gate(Gate::Rx(1.4), &[1]);
    let obs = [PauliTerm::z(0)];
    assert_eq!(run_spd(&circuit, 0.0, 0).unwrap().t_count, 5);
    assert_eq!(
        run_spd_observable(&circuit, &obs, 0.0, 0).unwrap().t_count,
        5
    );
    assert_eq!(
        run_spd_observable_light_cone(&circuit, &obs, 0.0, 0)
            .unwrap()
            .t_count,
        5
    );
    assert_eq!(run_spp(&circuit, 16, 42).unwrap().t_count, 5);
}

fn pauli_string_rotation(theta: f64, letters: &[(usize, PauliAxis)]) -> (Gate, Vec<usize>) {
    let targets = letters.iter().map(|&(q, _)| q).collect();
    let axes = letters.iter().map(|&(_, axis)| axis).collect();
    (
        Gate::PauliRot(Box::new(crate::gates::PauliRotData { theta, axes })),
        targets,
    )
}

/// Clifford layers between `PauliRot` strings of weight 1 to 5 on shuffled targets,
/// with angles drawn from `(-pi, pi)` and pinned near `0`, `pi/2` and `+-pi`.
fn random_pauli_string_circuit(n: usize, rotations: usize, seed: u64) -> Circuit {
    use rand::RngExt;
    use rand::seq::SliceRandom;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let edge_angles = [
        std::f64::consts::PI - 1e-3,
        -std::f64::consts::PI + 1e-7,
        std::f64::consts::PI,
        std::f64::consts::FRAC_PI_2,
        -2e-9,
    ];
    let letters = [PauliAxis::X, PauliAxis::Y, PauliAxis::Z];
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::SX, Gate::X];
    let mut c = Circuit::new(n, 0);
    let mut qubits: Vec<usize> = (0..n).collect();
    for k in 0..rotations {
        for q in 0..n {
            c.add_gate(
                cliffords[rng.random_range(0..cliffords.len())].clone(),
                &[q],
            );
        }
        qubits.shuffle(&mut rng);
        c.add_gate(Gate::Cx, &[qubits[0], qubits[1]]);
        let weight = 1 + k % 5.min(n);
        qubits.shuffle(&mut rng);
        let string: Vec<(usize, PauliAxis)> = qubits[..weight]
            .iter()
            .map(|&q| (q, letters[rng.random_range(0..3)]))
            .collect();
        let theta = if k % 4 == 3 {
            edge_angles[(k / 4) % edge_angles.len()]
        } else {
            rng.random_range(-std::f64::consts::PI..std::f64::consts::PI)
        };
        let (gate, targets) = pauli_string_rotation(theta, &string);
        c.add_gate(gate, &targets);
        if k % 3 == 1 {
            c.add_gate(Gate::Rx(rng.random_range(-1.0..1.0)), &[qubits[0]]);
        }
    }
    c
}

#[test]
fn pauli_string_rotations_match_the_statevector_and_their_lowering() {
    let n = 6;
    let mut observables: Vec<Vec<PauliTerm>> = (0..n)
        .flat_map(|q| [PauliTerm::x(q), PauliTerm::y(q), PauliTerm::z(q)])
        .map(|term| vec![term])
        .collect();
    observables.push(vec![PauliTerm::x(0), PauliTerm::y(2), PauliTerm::z(5)]);
    observables.push(vec![PauliTerm::y(1), PauliTerm::y(4)]);
    let mut nonzero = 0;
    for seed in 0..12u64 {
        let circuit = random_pauli_string_circuit(n, 15, 42 + seed);
        let lowered = crate::circuit::expand_pauli_rotations(&circuit).into_owned();
        assert!(!lowered.instructions.iter().any(|inst| matches!(
            inst,
            Instruction::Gate {
                gate: Gate::PauliRot(_),
                ..
            }
        )));
        let want = reference_expectations(
            &circuit,
            crate::sim::BackendKind::Statevector,
            None,
            &observables,
        );
        let noise = NoiseModel::with_amplitude_damping(&circuit, 0.0);
        for (obs, want) in observables.iter().zip(&want) {
            let native = run_spd_observable(&circuit, obs, 0.0, 0).unwrap();
            let ladder = run_spd_observable(&lowered, obs, 0.0, 0).unwrap();
            let cone = run_spd_observable_light_cone(&circuit, obs, 0.0, 0)
                .unwrap()
                .mean;
            let path = run_pauli_path_observable(&circuit, &noise, obs, 0.0, 0)
                .unwrap()
                .mean;
            for (label, got) in [
                ("spd", native.mean),
                ("ladder", ladder.mean),
                ("light cone", cone),
                ("path", path),
            ] {
                assert!(
                    (got - want).abs() < 1e-10,
                    "seed {seed} {label} {obs:?}: {got} vs {want}"
                );
            }
            assert_eq!(native.peak_terms, ladder.peak_terms, "seed {seed} {obs:?}");
            nonzero += usize::from(want.abs() > 1e-3);
        }
        let marginals = run_spd(&circuit, 0.0, 0).unwrap();
        let ladder_marginals = run_spd(&lowered, 0.0, 0).unwrap();
        assert_eq!(marginals.t_count, ladder_marginals.t_count);
        assert_eq!(marginals.max_terms, ladder_marginals.max_terms);
        for q in 0..n {
            let want = want[3 * q + 2];
            let got = marginals.expectations[q];
            assert!(
                (got - want).abs() < 1e-10,
                "seed {seed} qubit {q}: {got} vs {want}"
            );
        }
    }
    assert!(nonzero > 40, "only {nonzero} observables away from zero");
}

#[test]
fn pauli_string_rotations_match_the_density_matrix_under_noise() {
    let circuit = random_pauli_string_circuit(4, 10, 7);
    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.04);
    let observables = native_axis_observables();
    let want = reference_expectations(
        &circuit,
        crate::sim::BackendKind::DensityMatrix,
        Some(&noise),
        &observables,
    );
    for (obs, want) in observables.iter().zip(&want) {
        let got = run_pauli_path_observable(&circuit, &noise, obs, 0.0, 0)
            .unwrap()
            .mean;
        assert!((got - want).abs() < 1e-12, "{obs:?}: {got} vs {want}");
    }
    assert!(want.iter().filter(|v| v.abs() > 1e-3).count() > 4);
}

#[test]
fn consecutive_pauli_string_rotations_merge_each_flip_into_a_term_already_present() {
    let (a, b) = (0.37, -1.1);
    let string = [(0, PauliAxis::Y), (1, PauliAxis::X), (2, PauliAxis::Z)];
    let mut split = Circuit::new(3, 0);
    let mut joined = Circuit::new(3, 0);
    for (circuit, angles) in [(&mut split, &[a, b][..]), (&mut joined, &[a + b][..])] {
        for &theta in angles {
            let (gate, targets) = pauli_string_rotation(theta, &string);
            circuit.add_gate(gate, &targets);
        }
    }
    for obs in [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::y(1), PauliTerm::z(2)],
        vec![PauliTerm::z(0), PauliTerm::z(1), PauliTerm::y(2)],
    ] {
        let split = run_spd_observable(&split, &obs, 0.0, 0).unwrap();
        let joined = run_spd_observable(&joined, &obs, 0.0, 0).unwrap();
        assert_eq!(split.peak_terms, 2, "{obs:?}");
        assert!((split.mean - joined.mean).abs() < 1e-12, "{obs:?}");
    }
    // The flip of Z_0 carries an X letter, so only the cos branch survives on |000>.
    let z = run_spd_observable(&split, &[PauliTerm::z(0)], 0.0, 0).unwrap();
    assert!((z.mean - (a + b).cos()).abs() < 1e-12);
}

#[test]
fn pauli_string_rotations_agree_across_key_widths() {
    let layouts: [(usize, [usize; 5]); 3] = [
        (5, [0, 1, 2, 3, 4]),
        (128, [3, 63, 64, 100, 127]),
        (200, [63, 64, 128, 150, 199]),
    ];
    let small = random_pauli_string_circuit(5, 12, 11);
    let mut reference: Option<[f64; 4]> = None;
    for (n, qubits) in layouts {
        let mut circuit = Circuit::new(n, 0);
        for inst in &small.instructions {
            let Instruction::Gate { gate, targets } = inst else {
                unreachable!()
            };
            let mapped: Vec<usize> = targets.iter().map(|&q| qubits[q]).collect();
            circuit.add_gate(gate.clone(), &mapped);
        }
        let obs = [PauliTerm::x(qubits[1]), PauliTerm::z(qubits[3])];
        let exact = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap().mean;
        let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0)
            .unwrap()
            .mean;
        let marginals = run_spd(&circuit, 0.0, 0).unwrap().expectations;
        let values = [exact, cone, marginals[qubits[0]], marginals[qubits[4]]];
        match reference {
            None => reference = Some(values),
            Some(expected) => {
                for (got, want) in values.iter().zip(expected) {
                    assert!((got - want).abs() < 1e-12, "{n} qubits: {got} vs {want}");
                }
            }
        }
    }
    assert!(reference.unwrap().iter().all(|v| v.abs() > 1e-6));
}

#[test]
fn a_pauli_string_rotation_counts_as_one_branching_gate() {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::H, &[0]);
    let (gate, targets) = pauli_string_rotation(
        0.4,
        &[(0, PauliAxis::X), (2, PauliAxis::Y), (3, PauliAxis::Z)],
    );
    circuit.add_gate(gate, &targets);
    circuit.add_gate(Gate::T, &[1]);
    let obs = [PauliTerm::z(0)];
    assert_eq!(run_spd(&circuit, 0.0, 0).unwrap().t_count, 2);
    assert_eq!(
        run_spd_observable(&circuit, &obs, 0.0, 0).unwrap().t_count,
        2
    );
    assert_eq!(run_spp(&circuit, 16, 42).unwrap().t_count, 2);
}

#[test]
fn spp_pauli_string_rotations_draw_the_samples_of_their_lowering() {
    let n = 6;
    let observables = [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(3)],
        vec![PauliTerm::y(1), PauliTerm::z(4)],
        vec![PauliTerm::x(0), PauliTerm::y(2), PauliTerm::z(5)],
    ];
    let mut branched = 0;
    for seed in 0..8u64 {
        let circuit = random_pauli_string_circuit(n, 15, 42 + seed);
        let lowered = crate::circuit::expand_pauli_rotations(&circuit).into_owned();
        for obs in &observables {
            let native = run_spp_observable(&circuit, obs, 2_000, seed).unwrap();
            let ladder = run_spp_observable(&lowered, obs, 2_000, seed).unwrap();
            assert_eq!(
                (native.mean, native.variance, native.nonzero_fraction),
                (ladder.mean, ladder.variance, ladder.nonzero_fraction),
                "seed {seed} {obs:?}"
            );
            assert_eq!(native.t_count, ladder.t_count);
            branched += usize::from(native.variance > 1e-6);
        }
        let native = run_spp(&circuit, 500, seed).unwrap();
        let ladder = run_spp(&lowered, 500, seed).unwrap();
        assert_eq!(native.expectations, ladder.expectations, "seed {seed}");
        assert_eq!(native.std_errors, ladder.std_errors, "seed {seed}");
        assert_eq!(native.t_count, ladder.t_count);
    }
    assert!(branched > 16, "only {branched} runs branched");
}

#[test]
fn spp_pauli_string_rotations_converge_to_the_statevector() {
    let mut nonzero = 0;
    for (n, rotations, seed) in [
        (4, 10, 100),
        (5, 10, 101),
        (5, 8, 103),
        (6, 10, 104),
        (6, 12, 105),
    ] {
        let circuit = random_pauli_string_circuit(n, rotations, seed);
        let mut observables: Vec<Vec<PauliTerm>> = (0..n)
            .flat_map(|q| [PauliTerm::x(q), PauliTerm::y(q), PauliTerm::z(q)])
            .map(|term| vec![term])
            .collect();
        observables.push(vec![PauliTerm::x(0), PauliTerm::y(2), PauliTerm::z(n - 1)]);
        let want = reference_expectations(
            &circuit,
            crate::sim::BackendKind::Statevector,
            None,
            &observables,
        );
        for (obs, want) in observables.iter().zip(&want) {
            let got = run_spp_observable(&circuit, obs, 20_000, 42).unwrap();
            assert!(
                (got.mean - want).abs() <= 5.0 * got.std_error + 2e-3,
                "{n}q seed {seed} {obs:?}: {} +- {} vs {want}",
                got.mean,
                got.std_error
            );
            nonzero += usize::from(want.abs() > 0.05);
        }
    }
    assert!(nonzero > 20, "only {nonzero} observables away from zero");
}

#[test]
fn spp_pauli_string_rotations_agree_across_key_widths() {
    let layouts: [(usize, [usize; 5]); 3] = [
        (5, [0, 1, 2, 3, 4]),
        (128, [3, 63, 64, 100, 127]),
        (200, [63, 64, 128, 150, 199]),
    ];
    let small = random_pauli_string_circuit(5, 12, 11);
    let mut reference = None;
    for (n, qubits) in layouts {
        let mut circuit = Circuit::new(n, 0);
        for inst in &small.instructions {
            let Instruction::Gate { gate, targets } = inst else {
                unreachable!()
            };
            let mapped: Vec<usize> = targets.iter().map(|&q| qubits[q]).collect();
            circuit.add_gate(gate.clone(), &mapped);
        }
        let lowered = crate::circuit::expand_pauli_rotations(&circuit).into_owned();
        let obs = [PauliTerm::x(qubits[1]), PauliTerm::z(qubits[3])];
        let native = run_spp_observable(&circuit, &obs, 4_000, 9).unwrap();
        let ladder = run_spp_observable(&lowered, &obs, 4_000, 9).unwrap();
        let values = (native.mean, native.variance);
        assert_eq!(values, (ladder.mean, ladder.variance), "{n} qubits");
        match reference {
            None => reference = Some(values),
            Some(expected) => assert_eq!(values, expected, "{n} qubits"),
        }
    }
    assert!(reference.unwrap().1 > 1e-6);
}

#[test]
fn clifford_angle_rotations_produce_no_ghost_terms_across_engines() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    let angles = [
        PI,
        -PI,
        FRAC_PI_2,
        -FRAC_PI_2,
        3.0 * FRAC_PI_2,
        TAU,
        PI + 1e-13,
        FRAC_PI_2 - 1e-13,
    ];
    let mut circuit = Circuit::new(3, 0);
    for q in 0..3 {
        circuit.add_gate(Gate::H, &[q]);
    }
    for i in 0..24 {
        let theta = angles[i % angles.len()];
        match i % 7 {
            0 => circuit.add_gate(Gate::Rz(theta), &[i % 3]),
            1 => circuit.add_gate(Gate::P(theta), &[i % 3]),
            2 => circuit.add_gate(Gate::Rzz(theta), &[i % 3, (i + 1) % 3]),
            3 => circuit.add_gate(Gate::Rx(theta), &[i % 3]),
            4 => circuit.add_gate(Gate::Ry(theta), &[i % 3]),
            5 => {
                let (gate, targets) = pauli_string_rotation(
                    theta,
                    &[(0, PauliAxis::X), (1, PauliAxis::Y), (2, PauliAxis::Z)],
                );
                circuit.add_gate(gate, &targets);
            }
            _ => circuit.add_gate(u3(theta, FRAC_PI_2, -FRAC_PI_2), &[i % 3]),
        }
    }
    for q in 0..3 {
        circuit.add_gate(Gate::H, &[q]);
    }

    let obs = [PauliTerm::z(0), PauliTerm::x(1), PauliTerm::y(2)];
    let sv_want = reference_expectations(
        &circuit,
        crate::sim::BackendKind::Statevector,
        None,
        &[
            vec![PauliTerm::z(0)],
            vec![PauliTerm::z(1)],
            vec![PauliTerm::z(2)],
            obs.to_vec(),
        ],
    );

    let spd = run_spd(&circuit, 0.0, 0).unwrap();
    assert_eq!(spd.max_terms, 1);
    for (&got, &want) in spd.expectations.iter().zip(&sv_want[..3]) {
        assert!((got - want).abs() < 1e-11);
    }

    let spd_obs = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
    assert_eq!(spd_obs.peak_terms, 1);
    assert!((spd_obs.mean - sv_want[3]).abs() < 1e-11);

    let cone_obs = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
    assert_eq!(cone_obs.peak_terms, 1);
    assert!((cone_obs.mean - sv_want[3]).abs() < 1e-11);

    let unital_noise = NoiseModel::uniform_depolarizing(&circuit, 0.03);
    let dm_unital = reference_expectations(
        &circuit,
        crate::sim::BackendKind::DensityMatrix,
        Some(&unital_noise),
        &[obs.to_vec()],
    )[0];
    let path_unital = run_pauli_path_observable(&circuit, &unital_noise, &obs, 0.0, 0).unwrap();
    assert_eq!(path_unital.peak_terms, 1);
    assert!((path_unital.mean - dm_unital).abs() < 1e-11);

    let damp_noise = NoiseModel::with_amplitude_damping(&circuit, 0.03);
    let dm_damp = reference_expectations(
        &circuit,
        crate::sim::BackendKind::DensityMatrix,
        Some(&damp_noise),
        &[obs.to_vec()],
    )[0];
    let path_damp = run_pauli_path_observable(&circuit, &damp_noise, &obs, 0.0, 0).unwrap();
    assert!((path_damp.mean - dm_damp).abs() < 1e-11);

    // Register excess rotation path (`2 * theta` hits `pi/2` and `pi`).
    for theta in [FRAC_PI_4, FRAC_PI_2] {
        let terms = propagate_register_rotations(1, 0, [(0, 1, 0, theta)].into_iter());
        assert_eq!(terms.len(), 1);
    }
}

#[test]
fn auto_marginals_uses_spd_exact_for_clifford_angle_rotations() {
    use std::f64::consts::{FRAC_PI_2, PI};

    // Exceed MAX_GAUGE_QUBITS (64) so Auto routes past CliffordRegister to MarginalRoute::Spd.
    let n = 65;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for i in 0..24 {
        let q = i % n;
        let q_next = (i + 1) % n;
        circuit.add_gate(Gate::Rz(PI), &[q]);
        circuit.add_gate(Gate::Rzz(FRAC_PI_2), &[q, q_next]);
    }
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }

    let result = crate::sim::simulate(&circuit).seed(42).marginals().unwrap();
    assert_eq!(
        result.metadata.backend,
        crate::sim::ResolvedBackend::DeterministicPauli
    );
    assert_eq!(result.metadata.reason, crate::sim::RouteRule::SpdExact);
}

#[test]
fn spp_clifford_angles_draw_zero_rng_and_preserve_unit_norm() {
    use std::f64::consts::{FRAC_PI_2, PI};

    for theta in [
        PI,
        -PI,
        FRAC_PI_2,
        -FRAC_PI_2,
        PI + 1e-13,
        FRAC_PI_2 - 1e-13,
    ] {
        let b = RotBranch::new(theta);
        assert!(
            b.p_keep == 1.0 || b.p_keep == 0.0,
            "theta={theta}: {}",
            b.p_keep
        );
        if b.p_keep == 1.0 {
            assert_eq!(b.keep_weight.abs(), 1.0);
        } else {
            assert_eq!(b.flip_weight_im.abs(), 1.0);
        }
    }

    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::Rz(PI), &[0]);
    circuit.add_gate(Gate::Rzz(FRAC_PI_2), &[0, 1]);
    let (p_pi, t_pi) = pauli_string_rotation(PI, &[(0, PauliAxis::X), (2, PauliAxis::Z)]);
    circuit.add_gate(p_pi, &t_pi);
    let (p_half, t_half) =
        pauli_string_rotation(FRAC_PI_2, &[(1, PauliAxis::Y), (2, PauliAxis::X)]);
    circuit.add_gate(p_half, &t_half);
    circuit.add_gate(Gate::H, &[0]);

    let ops = coalesce_cliffords(&circuit);
    let obs = PauliVec::z_on_qubit(1, 0);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let _ = backward_propagate_coalesced(&ops, &obs, &mut rng);
    assert_eq!(rng.get_word_pos(), 0);

    let spp = run_spp_observable(&circuit, &[PauliTerm::z(0)], 64, 42).unwrap();
    assert_eq!(spp.variance, 0.0);
}

#[test]
fn small_non_clifford_angles_outside_identity_eps_still_branch() {
    use std::f64::consts::PI;

    for theta in [1e-10, PI + 1e-10] {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Rz(theta), &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let spd = run_spd_observable(&circuit, &[PauliTerm::z(0)], 0.0, 0).unwrap();
        assert_eq!(spd.peak_terms, 2, "theta={theta}");

        let b = RotBranch::new(theta);
        assert!(
            b.p_keep > 0.0 && b.p_keep < 1.0,
            "theta={theta}: {}",
            b.p_keep
        );
    }
}

#[test]
fn prepared_spd_matches_light_cone_and_statevector_across_observables() {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Rz(PI), &[0]);
    circuit.add_gate(Gate::Rx(0.41), &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Ry(FRAC_PI_2), &[1]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(u3(0.35, -0.7, 1.2), &[2]);
    circuit.add_gate(Gate::cu(Gate::Rz(0.8).matrix_2x2()), &[2, 3]);
    let (p_gate, p_targets) = pauli_string_rotation(TAU, &[(0, PauliAxis::X), (3, PauliAxis::Y)]);
    circuit.add_gate(p_gate, &p_targets);
    let (p_gate2, p_targets2) =
        pauli_string_rotation(-0.37, &[(1, PauliAxis::Z), (2, PauliAxis::X)]);
    circuit.add_gate(p_gate2, &p_targets2);

    let observables = vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1), PauliTerm::z(2)],
        vec![PauliTerm::y(0), PauliTerm::z(3)],
        vec![PauliTerm::z(2), PauliTerm::x(3)],
    ];
    let want = reference_expectations(
        &circuit,
        crate::sim::BackendKind::Statevector,
        None,
        &observables,
    );

    let prepared = PreparedSpd::new(&circuit, "light-cone SPD observable").unwrap();
    for (obs, sv_mean) in observables.iter().zip(want) {
        let from_prepared = prepared.eval_light_cone(obs, 1e-10, 16_384).unwrap();
        let direct = run_spd_observable_light_cone(&circuit, obs, 1e-10, 16_384).unwrap();
        assert_eq!(from_prepared.mean.to_bits(), direct.mean.to_bits());
        assert_eq!(from_prepared.t_count, direct.t_count);
        assert_eq!(from_prepared.peak_terms, direct.peak_terms);
        assert_eq!(
            from_prepared.total_discarded.to_bits(),
            direct.total_discarded.to_bits()
        );
        assert!((from_prepared.mean - sv_mean).abs() < 1e-9);
    }
}

#[test]
fn inverse_light_cone_bitset_handles_multi_word_empty_duplicate_and_oob() {
    for n in [0usize, 63, 64, 65, 128, 130] {
        let mut circuit = Circuit::new(n, 0);
        if n >= 2 {
            let last = n - 1;
            circuit.add_gate(Gate::H, &[0]);
            circuit.add_gate(Gate::T, &[last]);
            circuit.add_gate(Gate::Cx, &[0, last]);
            if n > 64 {
                circuit.add_gate(Gate::H, &[64]);
                circuit.add_gate(Gate::T, &[64]);
            }

            assert_eq!(
                inverse_light_cone(&circuit, &[]),
                vec![false; circuit.instructions.len()]
            );

            let cone_last = inverse_light_cone(&circuit, &[PauliTerm::z(last)]);
            let cone_dup = inverse_light_cone(&circuit, &[PauliTerm::z(last), PauliTerm::z(last)]);
            assert_eq!(cone_last, cone_dup);
            assert!(cone_last[0] && cone_last[1] && cone_last[2]);
            if n > 64 && last != 64 {
                assert!(!cone_last[3] && !cone_last[4]);
            }
        } else {
            assert_eq!(inverse_light_cone(&circuit, &[]), Vec::<bool>::new());
        }

        // Out-of-range observable qubit must not panic.
        let oob = inverse_light_cone(&circuit, &[PauliTerm::z(n + 100)]);
        assert_eq!(oob, vec![false; circuit.instructions.len()]);
    }
}

#[test]
fn spp_non_finite_rotation_angles_do_not_panic() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // Anticommuting Rz(bad)
        let mut rz_circuit = Circuit::new(1, 0);
        rz_circuit.add_gate(Gate::H, &[0]);
        rz_circuit.add_gate(Gate::Rz(bad), &[0]);
        rz_circuit.add_gate(Gate::H, &[0]);

        let rz_obs = run_spp_observable(&rz_circuit, &[PauliTerm::z(0)], 16, 42).unwrap();
        assert!(
            rz_obs.mean.is_nan(),
            "Rz({bad}) observable mean={}",
            rz_obs.mean
        );
        let rz_marg = run_spp(&rz_circuit, 16, 42).unwrap();
        assert!(
            rz_marg.expectations[0].is_nan(),
            "Rz({bad}) expectation={}",
            rz_marg.expectations[0]
        );

        // Commuting Rz(bad)
        let mut rz_comm = Circuit::new(1, 0);
        rz_comm.add_gate(Gate::Rz(bad), &[0]);
        let comm_obs = run_spp_observable(&rz_comm, &[PauliTerm::z(0)], 16, 42).unwrap();
        assert_eq!(comm_obs.mean, 1.0);
        let comm_marg = run_spp(&rz_comm, 16, 42).unwrap();
        assert_eq!(comm_marg.expectations[0], 1.0);

        // Anticommuting PauliRot(bad)
        let mut pr_circuit = Circuit::new(2, 0);
        let (pr_gate, pr_targets) =
            pauli_string_rotation(bad, &[(0, PauliAxis::X), (1, PauliAxis::Z)]);
        pr_circuit.add_gate(pr_gate, &pr_targets);

        let pr_obs = run_spp_observable(&pr_circuit, &[PauliTerm::z(0)], 16, 42).unwrap();
        assert!(
            pr_obs.mean.is_nan(),
            "PauliRot({bad}) observable mean={}",
            pr_obs.mean
        );
        let pr_marg = run_spp(&pr_circuit, 16, 42).unwrap();
        assert!(
            pr_marg.expectations[0].is_nan(),
            "PauliRot({bad}) expectation={}",
            pr_marg.expectations[0]
        );
        assert_eq!(pr_marg.expectations[1], 1.0);

        // Anticommuting Rzz(bad), Rx(bad), Ry(bad), P(bad)
        let mut mixed = Circuit::new(2, 0);
        mixed.add_gate(Gate::H, &[0]);
        mixed.add_gate(Gate::Rzz(bad), &[0, 1]);
        mixed.add_gate(Gate::Rx(bad), &[1]);
        mixed.add_gate(Gate::Ry(bad), &[0]);
        mixed.add_gate(Gate::P(bad), &[1]);
        let mixed_obs = run_spp_observable(&mixed, &[PauliTerm::z(0)], 16, 42).unwrap();
        assert!(mixed_obs.mean.is_nan());
        let mixed_marg = run_spp(&mixed, 16, 42).unwrap();
        assert!(mixed_marg.expectations.iter().all(|v| v.is_nan()));
    }
}

fn spp_multi_gate_circuit(n: usize, qubits: &[usize; 5]) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[qubits[0]]);
    c.add_gate(Gate::S, &[qubits[1]]);
    c.add_gate(Gate::Cx, &[qubits[0], qubits[2]]);
    c.add_gate(Gate::Cz, &[qubits[1], qubits[3]]);
    c.add_gate(Gate::Rz(0.37), &[qubits[0]]);
    c.add_gate(Gate::Rzz(-0.43), &[qubits[1], qubits[4]]);
    let (p_gate, p_targets) = pauli_string_rotation(
        0.61,
        &[
            (qubits[0], PauliAxis::X),
            (qubits[2], PauliAxis::Y),
            (qubits[4], PauliAxis::Z),
        ],
    );
    c.add_gate(p_gate, &p_targets);
    c.add_gate(Gate::H, &[qubits[2]]);
    c.add_gate(Gate::Rz(-0.29), &[qubits[3]]);
    c.add_gate(Gate::Cx, &[qubits[3], qubits[4]]);
    c
}

#[test]
fn spp_scratch_reuse_matches_fresh_backward_propagation() {
    for (n, qubits, obs_terms) in [
        (
            5usize,
            [0usize, 1, 2, 3, 4],
            vec![PauliTerm::x(0), PauliTerm::y(2), PauliTerm::z(4)],
        ),
        (
            130usize,
            [0usize, 63, 64, 65, 129],
            vec![PauliTerm::x(63), PauliTerm::y(64), PauliTerm::z(129)],
        ),
    ] {
        let circuit = spp_multi_gate_circuit(n, &qubits);
        let ops = coalesce_cliffords(&circuit);
        let (obs, _) = pauli_vec_from_terms(n, &obs_terms).unwrap();
        let mut scratch = PauliVec::new(obs.x.len());
        let mut rng_reuse = ChaCha8Rng::seed_from_u64(42);
        let mut rng_fresh = ChaCha8Rng::seed_from_u64(42);

        for sample in 0..200 {
            let w_reuse =
                backward_propagate_coalesced_into(&ops, &obs, &mut scratch, &mut rng_reuse);
            let (fresh, w_fresh) = backward_propagate_coalesced(&ops, &obs, &mut rng_fresh);
            assert_eq!(scratch.x, fresh.x, "n={n} sample={sample}");
            assert_eq!(scratch.z, fresh.z, "n={n} sample={sample}");
            assert_eq!(w_reuse.re.to_bits(), w_fresh.re.to_bits());
            assert_eq!(w_reuse.im.to_bits(), w_fresh.im.to_bits());
        }
    }
}

#[test]
fn spp_multi_word_reuse_does_not_leak_stale_bits_across_qubits_or_samples() {
    let small_qubits = [0usize, 1, 2, 3, 4];
    let wide_qubits = [0usize, 63, 64, 65, 129];

    let add_rx_ry = |c: &mut Circuit, q: &[usize; 5]| {
        c.add_gate(Gate::Rx(0.51), &[q[0]]);
        c.add_gate(Gate::Ry(-0.33), &[q[2]]);
        c.add_gate(Gate::H, &[q[4]]);
    };

    let mut small = spp_multi_gate_circuit(5, &small_qubits);
    add_rx_ry(&mut small, &small_qubits);
    let mut wide = spp_multi_gate_circuit(130, &wide_qubits);
    add_rx_ry(&mut wide, &wide_qubits);

    let obs_small = [PauliTerm::x(1), PauliTerm::y(2), PauliTerm::z(4)];
    let obs_wide = [PauliTerm::x(63), PauliTerm::y(64), PauliTerm::z(129)];
    let res_small = run_spp_observable(&small, &obs_small, 2000, 42).unwrap();
    let res_wide = run_spp_observable(&wide, &obs_wide, 2000, 42).unwrap();
    assert_eq!(res_small.mean.to_bits(), res_wide.mean.to_bits());
    assert_eq!(res_small.variance.to_bits(), res_wide.variance.to_bits());
    assert_eq!(
        res_small.nonzero_fraction.to_bits(),
        res_wide.nonzero_fraction.to_bits()
    );

    let spp_wide = run_spp(&wide, 8000, 42).unwrap();
    for untouched in [1usize, 62, 66, 128] {
        assert_eq!(spp_wide.expectations[untouched], 1.0);
        assert_eq!(spp_wide.std_errors[untouched], 0.0);
    }
    let spd_small = run_spd(&small, 0.0, 0).unwrap();
    for (i, &q) in wide_qubits.iter().enumerate() {
        let got = spp_wide.expectations[q];
        let want = spd_small.expectations[i];
        let tol = 5.0 * spp_wide.std_errors[q] + 1e-2;
        assert!(
            (got - want).abs() <= tol,
            "qubit {q}: got={got}, want={want}, tol={tol}"
        );
    }
}

#[test]
fn spp_scratch_reuse_preserves_clifford_snapping_and_zero_rng_draws() {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    let mut circuit = Circuit::new(70, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[64]);
    circuit.add_gate(Gate::Rz(PI + 1e-13), &[0]);
    circuit.add_gate(Gate::Rz(FRAC_PI_2 - 1e-13), &[64]);
    circuit.add_gate(Gate::Rzz(-2e-13), &[0, 64]);
    let (p_gate, p_targets) = pauli_string_rotation(
        TAU,
        &[(0, PauliAxis::X), (65, PauliAxis::Y), (69, PauliAxis::Z)],
    );
    circuit.add_gate(p_gate, &p_targets);
    circuit.add_gate(Gate::H, &[0]);

    let ops = coalesce_cliffords(&circuit);
    let (obs, _) = pauli_vec_from_terms(70, &[PauliTerm::z(0), PauliTerm::x(65)]).unwrap();
    let mut scratch = PauliVec::new(obs.x.len());
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        let _ = backward_propagate_coalesced_into(&ops, &obs, &mut scratch, &mut rng);
    }
    assert_eq!(rng.get_word_pos(), 0);

    let obs_res = run_spp_observable(&circuit, &[PauliTerm::z(0)], 64, 42).unwrap();
    assert_eq!(obs_res.variance, 0.0);
    let marg_res = run_spp(&circuit, 64, 42).unwrap();
    assert!(marg_res.std_errors.iter().all(|&se| se == 0.0));
}

#[test]
fn spp_scratch_reuse_isolates_non_finite_angles_across_samples_and_qubits() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut circuit = Circuit::new(70, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[64]);
        circuit.add_gate(Gate::Rz(bad), &[0]);
        circuit.add_gate(Gate::Rzz(bad), &[0, 64]);
        let (p_gate, p_targets) =
            pauli_string_rotation(bad, &[(0, PauliAxis::X), (64, PauliAxis::Z)]);
        circuit.add_gate(p_gate, &p_targets);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[64]);

        let res = run_spp(&circuit, 32, 42).unwrap();
        assert!(res.expectations[0].is_nan());
        assert!(res.expectations[64].is_nan());
        assert_eq!(res.expectations[65], 1.0);
        assert_eq!(res.std_errors[65], 0.0);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn spp_parallel_matches_sequential_thread_counts() {
    let mut circuit = Circuit::new(70, 0);
    for q in [0usize, 1, 31, 63, 64, 69] {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::Cx, &[0, 64]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Rz(0.31), &[63]);
    circuit.add_gate(Gate::Rzz(-0.55), &[1, 69]);
    let (p_gate, p_targets) =
        pauli_string_rotation(0.47, &[(31, PauliAxis::X), (64, PauliAxis::Y)]);
    circuit.add_gate(p_gate, &p_targets);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::H, &[64]);

    let one = crate::ThreadPool::with_threads(1)
        .unwrap()
        .install(|| run_spp(&circuit, 1024, 42).unwrap());
    let four = crate::ThreadPool::with_threads(4)
        .unwrap()
        .install(|| run_spp(&circuit, 1024, 42).unwrap());

    assert_eq!(one.expectations.len(), four.expectations.len());
    for (a, b) in one.expectations.iter().zip(&four.expectations) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in one.std_errors.iter().zip(&four.std_errors) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(
        one.nonzero_fraction.to_bits(),
        four.nonzero_fraction.to_bits()
    );
}
