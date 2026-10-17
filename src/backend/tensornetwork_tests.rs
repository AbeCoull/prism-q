use super::*;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Circuit;
use crate::gates::{Gate, MultiFusedData};

const EPS: f64 = 1e-10;

fn assert_probs_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < EPS,
            "prob[{i}]: TN={x}, expected={y}, diff={}",
            (x - y).abs()
        );
    }
}

#[test]
fn test_init_zero_state() {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(3, 0).unwrap();
    let probs = tn.probabilities().unwrap();
    assert_eq!(probs.len(), 8);
    assert!((probs[0] - 1.0).abs() < EPS);
    for &p in &probs[1..] {
        assert!(p.abs() < EPS);
    }
}

#[test]
fn test_single_qubit_h() {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(1, 0).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    let probs = tn.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < EPS);
    assert!((probs[1] - 0.5).abs() < EPS);
}

#[test]
fn test_single_qubit_x() {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(1, 0).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    let probs = tn.probabilities().unwrap();
    assert!(probs[0].abs() < EPS);
    assert!((probs[1] - 1.0).abs() < EPS);
}

#[test]
fn test_two_qubit_cx_bell() {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(2, 0).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: smallvec::smallvec![0, 1],
    })
    .unwrap();
    let probs = tn.probabilities().unwrap();
    assert!((probs[0] - 0.5).abs() < EPS);
    assert!(probs[1].abs() < EPS);
    assert!(probs[2].abs() < EPS);
    assert!((probs[3] - 0.5).abs() < EPS);
}

#[test]
fn test_parametric_rx() {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(1, 0).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::Rx(std::f64::consts::PI),
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    let probs = tn.probabilities().unwrap();
    assert!(probs[0].abs() < EPS);
    assert!((probs[1] - 1.0).abs() < EPS);
}

#[test]
fn test_measure_deterministic() {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(1, 1).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::X,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    tn.apply(&Instruction::Measure {
        qubit: 0,
        classical_bit: 0,
    })
    .unwrap();
    assert!(tn.classical_results()[0]);
}

#[test]
fn test_measure_seeded() {
    let run = |seed| {
        let mut tn = TensorNetworkBackend::new(seed);
        tn.init(1, 1).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        tn.apply(&Instruction::Measure {
            qubit: 0,
            classical_bit: 0,
        })
        .unwrap();
        tn.classical_results()[0]
    };
    let r1 = run(42);
    let r2 = run(42);
    assert_eq!(r1, r2);
}

#[test]
fn test_fused_gate() {
    let ht_mat = crate::gates::mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
    let mut tn_fused = TensorNetworkBackend::new(42);
    tn_fused.init(1, 0).unwrap();
    tn_fused
        .apply(&Instruction::Gate {
            gate: Gate::Fused(Box::new(ht_mat)),
            targets: smallvec::smallvec![0],
        })
        .unwrap();

    let mut tn_individual = TensorNetworkBackend::new(42);
    tn_individual.init(1, 0).unwrap();
    tn_individual
        .apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
    tn_individual
        .apply(&Instruction::Gate {
            gate: Gate::T,
            targets: smallvec::smallvec![0],
        })
        .unwrap();

    assert_probs_close(
        &tn_fused.probabilities().unwrap(),
        &tn_individual.probabilities().unwrap(),
    );
}

#[test]
fn test_multi_fused() {
    let h_mat = Gate::H.matrix_2x2();
    let t_mat = Gate::T.matrix_2x2();
    let x_mat = Gate::X.matrix_2x2();

    let mut tn_mf = TensorNetworkBackend::new(42);
    tn_mf.init(3, 0).unwrap();
    tn_mf
        .apply(&Instruction::Gate {
            gate: Gate::MultiFused(Box::new(MultiFusedData {
                gates: vec![(0, h_mat), (1, t_mat), (2, x_mat)],
                all_diagonal: false,
            })),
            targets: smallvec::smallvec![0, 1, 2],
        })
        .unwrap();

    let mut tn_ind = TensorNetworkBackend::new(42);
    tn_ind.init(3, 0).unwrap();
    tn_ind
        .apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
    tn_ind
        .apply(&Instruction::Gate {
            gate: Gate::T,
            targets: smallvec::smallvec![1],
        })
        .unwrap();
    tn_ind
        .apply(&Instruction::Gate {
            gate: Gate::X,
            targets: smallvec::smallvec![2],
        })
        .unwrap();

    assert_probs_close(
        &tn_mf.probabilities().unwrap(),
        &tn_ind.probabilities().unwrap(),
    );
}

#[test]
fn test_golden_vs_statevector() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(0.7), &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Rx(1.2), &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::H, &[3]);

    let mut sv = StatevectorBackend::new(42);
    sv.init(4, 0).unwrap();
    for inst in &c.instructions {
        sv.apply(inst).unwrap();
    }
    let sv_probs = sv.probabilities().unwrap();

    let mut tn = TensorNetworkBackend::new(42);
    tn.init(4, 0).unwrap();
    for inst in &c.instructions {
        tn.apply(inst).unwrap();
    }
    let tn_probs = tn.probabilities().unwrap();

    assert_probs_close(&tn_probs, &sv_probs);
}

#[test]
fn test_export_statevector() {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(2, 0).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: smallvec::smallvec![0, 1],
    })
    .unwrap();

    let sv = tn.export_statevector().unwrap();
    assert_eq!(sv.len(), 4);
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert!((sv[0].re - h).abs() < EPS);
    assert!(sv[1].norm() < EPS);
    assert!(sv[2].norm() < EPS);
    assert!((sv[3].re - h).abs() < EPS);
}

#[test]
fn test_cu_gate() {
    let rz_mat = Gate::Rz(0.5).matrix_2x2();

    let mut tn = TensorNetworkBackend::new(42);
    tn.init(2, 0).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::Cu(Box::new(rz_mat)),
        targets: smallvec::smallvec![0, 1],
    })
    .unwrap();

    let mut sv = StatevectorBackend::new(42);
    sv.init(2, 0).unwrap();
    sv.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    sv.apply(&Instruction::Gate {
        gate: Gate::Cu(Box::new(rz_mat)),
        targets: smallvec::smallvec![0, 1],
    })
    .unwrap();

    assert_probs_close(&tn.probabilities().unwrap(), &sv.probabilities().unwrap());
}

#[test]
fn test_scalar_expectation_matches_statevector() {
    let circuit = crate::circuits::hardware_efficient_ansatz(8, 2, 42);
    let terms = [PauliTerm::z(1), PauliTerm::x(5)];

    let expected = crate::sim::run_expectation_values(&circuit, &[terms.to_vec()], 42).unwrap()[0];
    let actual =
        expectation_zero_state(&circuit, &terms, &ResolvedLimits::process_default()).unwrap();
    assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
}

// Four brickwork layers confine Z on qubit 0 to the first few qubits, so
// the gates at the far end never enter the network.
#[test]
fn test_scalar_expectation_prunes_to_the_light_cone() {
    let circuit = crate::circuits::brickwork_circuit(14, 4, 42);
    let local = [PauliTerm::z(0)];
    let global: Vec<PauliTerm> = (0..14).map(PauliTerm::z).collect();

    let expected = crate::sim::run_expectation_values(&circuit, &[local.to_vec()], 42).unwrap()[0];
    let actual =
        expectation_zero_state(&circuit, &local, &ResolvedLimits::process_default()).unwrap();
    assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");

    let pruned = ScalarExpectationNetwork::for_observable(&circuit, &local).unwrap();
    let full = ScalarExpectationNetwork::for_observable(&circuit, &global).unwrap();
    assert!(
        2 * pruned.tensors.len() < full.tensors.len(),
        "{} tensors against {} over the whole circuit",
        pruned.tensors.len(),
        full.tensors.len()
    );
}

// Idle qubits leave one disconnected component each, which is what
// join_disjoint exists for.
#[test]
fn test_scalar_expectation_with_idle_qubits() {
    let mut circuit = Circuit::new(9, 0);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_gate(Gate::Ry(0.7), &[6]);
    let terms = [PauliTerm::z(2), PauliTerm::z(3), PauliTerm::x(6)];

    let expected = crate::sim::run_expectation_values(&circuit, &[terms.to_vec()], 42).unwrap()[0];
    let actual =
        expectation_zero_state(&circuit, &terms, &ResolvedLimits::process_default()).unwrap();
    assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
}

// Twelve brickwork layers under Z on every qubit plan lower fused, peak
// 262144 against 1048576, so the fused network is the one contracted.
#[test]
fn test_scalar_expectation_takes_the_fused_network_when_it_plans_lower() {
    let circuit = crate::circuits::brickwork_circuit(16, 12, 42);
    let terms: Vec<PauliTerm> = (0..16).map(PauliTerm::z).collect();

    let unfused = plan_with_restarts(&scalar_network(&circuit, &terms).tensors);
    let fused_circuit = crate::circuit::fusion::fuse_circuit(&circuit, true);
    let fused = plan_with_restarts(&scalar_network(&fused_circuit, &terms).tensors);
    assert!(unfused.total >= FUSED_PLAN_TOTAL);
    assert!((fused.peak, fused.total) < (unfused.peak, unfused.total));

    let expected =
        crate::sim::run_expectation_values(&circuit, std::slice::from_ref(&terms), 42).unwrap()[0];
    let actual =
        expectation_zero_state(&circuit, &terms, &ResolvedLimits::process_default()).unwrap();
    assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
}

fn scalar_network(circuit: &Circuit, terms: &[PauliTerm]) -> ScalarExpectationNetwork {
    ScalarExpectationNetwork::for_observable(circuit, terms).unwrap()
}

// hardware_efficient_ansatz(30, 7) is a recorded case of the greedy tree
// peaking at 16.8M elements, past RESTART_PEAK_THRESHOLD, so the restart
// arm runs. Planning walks metadata only, so no contraction executes here.
#[test]
fn transpose_matches_an_index_by_index_permutation() {
    let cases: [(&[usize], &[usize]); 6] = [
        (&[2, 3, 4], &[2, 0, 1]),
        (&[5, 1, 3], &[1, 2, 0]),
        (&[64, 2, 2, 2, 2, 16], &[1, 2, 0, 3, 4, 5]),
        (&[16, 2, 4, 2, 64, 2], &[3, 1, 5, 0, 2, 4]),
        (&[8, 2, 2, 8, 2, 2, 2, 16], &[1, 7, 2, 4, 0, 6, 3, 5]),
        (&[3, 16, 2, 2, 7, 32], &[4, 0, 2, 5, 3, 1]),
    ];
    for (shape, perm) in cases {
        let total: usize = shape.iter().product();
        let tensor = Tensor {
            data: (0..total)
                .map(|i| Complex64::new(i as f64, -(i as f64)))
                .collect(),
            shape: SmallVec::from_slice(shape),
            legs: (0..shape.len()).collect(),
        };
        let out = transpose(&tensor, perm);
        let mut index: SmallVec<[usize; 8]> = SmallVec::from_elem(0, shape.len());
        for (flat, value) in out.data.iter().enumerate() {
            let mut rem = flat;
            for axis in (0..perm.len()).rev() {
                index[perm[axis]] = rem % shape[perm[axis]];
                rem /= shape[perm[axis]];
            }
            let src = index.iter().zip(shape).fold(0, |acc, (&i, &e)| acc * e + i);
            assert_eq!(
                value.re, src as f64,
                "shape {shape:?} perm {perm:?} at {flat}"
            );
        }
        assert_eq!(out.legs.as_slice(), perm);
    }
}

#[test]
fn test_plan_restarts_deterministic_and_never_worse() {
    let circuit = crate::circuits::hardware_efficient_ansatz(30, 7, 42);
    let terms = [PauliTerm::z(0), PauliTerm::z(15)];
    let network = scalar_network(&circuit, &terms);
    let slots: Vec<Option<TensorMeta>> = network
        .tensors
        .iter()
        .map(|t| Some(TensorMeta::of(t)))
        .collect();

    let greedy = plan_pairs(slots, None, usize::MAX).unwrap();
    assert!(
        greedy.peak >= RESTART_PEAK_THRESHOLD,
        "fixture no longer reaches the restart arm: greedy peak {}",
        greedy.peak
    );

    let best_a = plan_with_restarts(&network.tensors);
    let best_b = plan_with_restarts(&network.tensors);
    assert_eq!(best_a.pairs, best_b.pairs);
    assert!(best_a.peak <= greedy.peak);
    println!(
        "greedy peak {} restart peak {} ({} pairs)",
        greedy.peak,
        best_a.peak,
        best_a.pairs.len()
    );
}

#[test]
fn test_slices_sum_to_the_unsliced_contraction() {
    let circuit = crate::circuits::hardware_efficient_ansatz(12, 4, 42);
    let terms = [PauliTerm::z(0), PauliTerm::z(6)];
    let exact =
        expectation_zero_state(&circuit, &terms, &ResolvedLimits::process_default()).unwrap();

    let network = scalar_network(&circuit, &terms);
    let plan = plan_with_restarts(&network.tensors);
    let limits = ContractionLimits {
        peak_cap: 1 << 8,
        peak_source: LimitSource::Request,
        slice_budget: 1 << 10,
        tolerance: None,
    };
    assert!(
        plan.peak > limits.peak_cap,
        "fixture plans {} elements, already under the cap",
        plan.peak
    );

    let metas: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();
    let slice = choose_slices(&metas, &plan, limits).expect("the fixture slices");
    assert!(slice.count > 1, "the fixture did not slice");
    assert!(slice.peak <= limits.peak_cap);
    // One slice per wave, three per wave, and every slice in one wave.
    for cap in [slice.peak, slice.peak * 3, usize::MAX] {
        let summed = contract_slices(&network.tensors, &plan, &slice, cap);
        assert!(
            (summed.data[0].re - exact).abs() < 1e-10,
            "{} vs {exact} over {} slices at cap {cap}",
            summed.data[0].re,
            slice.count
        );
    }
}

#[test]
fn test_slice_choice_refuses_what_the_budget_cannot_reach() {
    let circuit = crate::circuits::hardware_efficient_ansatz(12, 4, 42);
    let terms = [PauliTerm::z(0), PauliTerm::z(6)];
    let network = scalar_network(&circuit, &terms);
    let plan = plan_with_restarts(&network.tensors);
    let metas: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();

    let limits = ContractionLimits {
        peak_cap: 1 << 8,
        peak_source: LimitSource::Request,
        slice_budget: 2,
        tolerance: None,
    };
    assert!(choose_slices(&metas, &plan, limits).is_none());
}

#[test]
fn test_pinned_axes_keep_their_extent_at_one() {
    let tensor = Tensor {
        data: (0..6).map(|i| Complex64::new(i as f64, 0.0)).collect(),
        shape: smallvec::smallvec![2, 3],
        legs: smallvec::smallvec![7, 8],
    };

    let rows = pin_axes(&tensor, &[(0, 1)]);
    assert_eq!(rows.shape.as_slice(), &[1, 3]);
    assert_eq!(rows.data, vec_of(&[3.0, 4.0, 5.0]));

    let cols = pin_axes(&tensor, &[(1, 2)]);
    assert_eq!(cols.shape.as_slice(), &[2, 1]);
    assert_eq!(cols.data, vec_of(&[2.0, 5.0]));

    let one = pin_axes(&tensor, &[(0, 1), (1, 0)]);
    assert_eq!(one.shape.as_slice(), &[1, 1]);
    assert_eq!(one.data, vec_of(&[3.0]));
}

#[test]
fn test_kept_rank_drops_only_what_the_budget_covers() {
    // Squared weights 1, 0.25, 0.01 and 0.0001, and the budget is an
    // absolute squared weight, not a fraction.
    let values = [1.0, 0.5, 0.1, 0.01];
    assert_eq!(kept_rank(&values, 0.0), 4);
    assert_eq!(kept_rank(&values, 0.0001), 3);
    assert_eq!(kept_rank(&values, 0.011), 2);
    assert_eq!(kept_rank(&values, 10.0), 1, "never fewer than one");
}

// A cut at zero tolerance keeps the full rank, so contracting the halves
// back has to reproduce the tensor: the split is exact and the only
// thing a tolerance buys is a shorter bond.
#[test]
fn test_a_zero_tolerance_cut_reconstructs_the_tensor() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let tensor = Tensor {
        data: (0..32)
            .map(|_| {
                use rand::RngExt;
                Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5)
            })
            .collect(),
        shape: smallvec::smallvec![2, 2, 2, 4],
        legs: smallvec::smallvec![10, 11, 12, 13],
    };

    let (far, near, discarded) = split_bond(&tensor, &[13], 99, 0.0).expect("the cut applies");
    assert_eq!(discarded, 0.0);
    assert!(far.legs.contains(&99) && near.legs.contains(&99));

    let rebuilt = contract(&far, &near);
    let perm: SmallVec<[usize; 6]> = tensor
        .legs
        .iter()
        .map(|leg| {
            rebuilt
                .legs
                .iter()
                .position(|held| held == leg)
                .expect("the halves carry every original leg")
        })
        .collect();
    let ordered = transpose(&rebuilt, &perm);
    assert_eq!(ordered.shape, tensor.shape);
    for (got, want) in ordered.data.iter().zip(&tensor.data) {
        assert!((got - want).norm() < 1e-12, "{got} vs {want}");
    }
}

// A cut that cannot shrink the blocked contraction declines, so the
// caller holds the tensor whole rather than paying for an SVD and two
// factors that buy nothing.
#[test]
fn test_a_cut_that_cannot_shrink_declines() {
    let tensor = Tensor {
        data: vec![Complex64::new(1.0, 0.0); 8],
        shape: smallvec::smallvec![2, 4],
        legs: smallvec::smallvec![10, 11],
    };
    assert!(
        split_bond(&tensor, &[10, 11], 99, 0.5).is_none(),
        "no far side"
    );
    assert!(split_bond(&tensor, &[], 99, 0.5).is_none(), "no near side");
}

fn vec_of(values: &[f64]) -> Vec<Complex64> {
    values.iter().map(|&v| Complex64::new(v, 0.0)).collect()
}

// Every noise stream must land on the same scalar: replay correctness for
// arbitrary plan orders is the risk the planner split introduces.
#[test]
fn test_noisy_plans_execute_to_the_same_scalar() {
    let circuit = crate::circuits::hardware_efficient_ansatz(8, 3, 42);
    let terms = [PauliTerm::z(0), PauliTerm::x(4)];
    let expected =
        expectation_zero_state(&circuit, &terms, &ResolvedLimits::process_default()).unwrap();

    for seed in 0..5u64 {
        let mut network = scalar_network(&circuit, &terms);
        let slots: Vec<Option<TensorMeta>> = network
            .tensors
            .iter()
            .map(|t| Some(TensorMeta::of(t)))
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let plan = plan_pairs(slots, Some((&mut rng, 1.0)), usize::MAX).unwrap();

        let result = replay_plan(&mut network.tensors, &plan);
        assert_eq!(result.data.len(), 1);
        assert!(
            (result.data[0].re - expected).abs() < EPS,
            "seed {seed}: {} vs {expected}",
            result.data[0].re
        );
    }
}

// Mid-circuit measure and reset must agree with the statevector on the
// seeded outcome stream, not just on the marginals, and on the final
// distribution after further gates.
#[test]
fn test_mid_circuit_measure_reset_matches_statevector() {
    let mut c = Circuit::new(5, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(0.7), &[2]);
    c.add_measure(1, 0);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::H, &[1]);
    c.add_reset(0);
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_measure(2, 1);
    c.add_gate(Gate::Ry(0.3), &[4]);

    for seed in [42u64, 7, 12345] {
        let mut sv = StatevectorBackend::new(seed);
        sv.init(5, 2).unwrap();
        let mut tn = TensorNetworkBackend::new(seed);
        tn.init(5, 2).unwrap();
        for inst in &c.instructions {
            sv.apply(inst).unwrap();
            tn.apply(inst).unwrap();
        }
        assert_eq!(
            tn.classical_results(),
            sv.classical_results(),
            "seed {seed}"
        );
        assert_probs_close(&tn.probabilities().unwrap(), &sv.probabilities().unwrap());
    }
}

// A measurement past the dense query ceiling must succeed and keep the
// deferred form: no rank-n tensor, and the network still answers
// expectation queries that never build a 2^n vector.
#[test]
fn test_measurement_past_dense_ceiling_keeps_network() {
    let n = 30;
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(n, 1).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: smallvec::smallvec![0, 1],
    })
    .unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: smallvec::smallvec![1, 2],
    })
    .unwrap();
    tn.apply(&Instruction::Measure {
        qubit: 1,
        classical_bit: 0,
    })
    .unwrap();

    assert!(tn.tensors.len() > 1);
    assert!(tn.tensors.iter().all(|t| t.rank() < 6));

    let outcome = tn.classical_results()[0];
    let expected = if outcome { -1.0 } else { 1.0 };
    let exps = tn
        .pauli_expectations(&[vec![PauliTerm::z(0)], vec![PauliTerm::z(2)]])
        .unwrap();
    assert!(
        (exps[0] - expected).abs() < EPS,
        "{} vs {expected}",
        exps[0]
    );
    assert!(
        (exps[1] - expected).abs() < EPS,
        "{} vs {expected}",
        exps[1]
    );
}

// Joint distribution check for the conditional sweep against the dense
// route with per-outcome binomial bands, plus the contract that sampling
// leaves the state untouched. Calls the sweep directly: the public path
// takes the dense arm at this width.
#[test]
fn test_native_sampling_matches_dense_distribution() {
    let circuit = crate::circuits::cz_chain_circuit(6, 3, 42);
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(6, 0).unwrap();
    for inst in &circuit.instructions {
        tn.apply(inst).unwrap();
    }
    let probs_before = tn.probabilities().unwrap();

    let shots = 2000usize;
    let samples = tn.sample_native(shots, 42).unwrap();

    let mut counts = vec![0usize; 1 << 6];
    for shot in 0..shots {
        let mut index = 0usize;
        for q in 0..6 {
            if samples.bit(shot, q) {
                index |= 1 << q;
            }
        }
        counts[index] += 1;
    }
    for (index, (&count, &p)) in counts.iter().zip(&probs_before).enumerate() {
        let freq = count as f64 / shots as f64;
        let sigma = (p * (1.0 - p) / shots as f64).sqrt().max(1e-3);
        assert!(
            (freq - p).abs() < 6.0 * sigma,
            "outcome {index}: {freq} vs {p}"
        );
    }

    assert_probs_close(&tn.probabilities().unwrap(), &probs_before);
}

// Per-qubit counts convergence where the dense route cannot answer at
// all: 30 independent rotations, marginals known analytically.
#[test]
fn test_native_sampling_past_dense_ceiling() {
    let n = 30;
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(n, 0).unwrap();
    for q in 0..n {
        tn.apply(&Instruction::Gate {
            gate: Gate::Ry(0.9),
            targets: smallvec::smallvec![q],
        })
        .unwrap();
    }

    let shots = 500usize;
    let samples = tn.sample_basis_states(shots, 42).unwrap();
    let p_one = (0.45f64).sin().powi(2);
    let sigma = (p_one * (1.0 - p_one) / shots as f64).sqrt();
    for q in [0usize, 7, 15, 29] {
        let count = (0..shots).filter(|&shot| samples.bit(shot, q)).count();
        let freq = count as f64 / shots as f64;
        assert!(
            (freq - p_one).abs() < 5.0 * sigma,
            "qubit {q}: {freq} vs {p_one}"
        );
    }
}

fn loaded_backend(circuit: &Circuit) -> TensorNetworkBackend {
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(circuit.num_qubits, 0).unwrap();
    for inst in &circuit.instructions {
        tn.apply(inst).unwrap();
    }
    tn
}

fn planner_calls() -> usize {
    PLANNER_CALLS.with(|calls| calls.get())
}

fn assert_plan_cache_transparent(circuit: &Circuit, shots: usize) {
    let mut tn = loaded_backend(circuit);
    let cached = tn.sample_native(shots, 42).unwrap();
    let uncached = tn.sample_sweep(shots, 42, None).unwrap();
    assert_eq!(cached.words, uncached.words);
}

#[test]
fn test_plan_cache_shots_match_uncached_sweep_on_chain() {
    assert_plan_cache_transparent(&crate::circuits::cz_chain_circuit(12, 4, 42), 8);
}

#[test]
fn test_plan_cache_shots_match_uncached_sweep_on_random_circuit() {
    assert_plan_cache_transparent(&crate::circuits::random_circuit(8, 5, 42), 8);
}

#[test]
fn test_plan_cache_recomputes_on_fingerprint_mismatch() {
    let circuit = crate::circuits::cz_chain_circuit(8, 3, 42);
    let mut tn = loaded_backend(&circuit);
    let mut plans: Vec<Option<CachedPlan>> = std::iter::repeat_with(|| None).take(8).collect();
    tn.sample_sweep(1, 42, Some(&mut plans)).unwrap();
    let genuine = plans[3].as_ref().unwrap().fingerprint;
    plans[3].as_mut().unwrap().fingerprint = !genuine;

    let before = planner_calls();
    tn.sample_sweep(1, 42, Some(&mut plans)).unwrap();
    assert_eq!(planner_calls() - before, 1);
    assert_eq!(plans[3].as_ref().unwrap().fingerprint, genuine);
}

fn leg_network(legs: &[[LegId; 2]]) -> Vec<Tensor> {
    legs.iter()
        .map(|pair| Tensor {
            data: vec![Complex64::new(1.0, 0.0); 4],
            shape: smallvec::smallvec![2, 2],
            legs: pair.iter().copied().collect(),
        })
        .collect()
}

#[test]
fn test_cached_plan_replans_when_only_leg_ids_differ() {
    let first = leg_network(&[[0, 1], [1, 2]]);
    let second = leg_network(&[[0, 1], [1, 3]]);
    let mut slot = None;
    let before = planner_calls();
    cached_plan(&first, &mut slot);
    let stored = slot.as_ref().unwrap().fingerprint;
    cached_plan(&first, &mut slot);
    assert_eq!(planner_calls() - before, 1);
    cached_plan(&second, &mut slot);
    assert_eq!(planner_calls() - before, 2);
    assert_ne!(slot.as_ref().unwrap().fingerprint, stored);
}

#[test]
fn test_plan_cache_plans_each_position_once() {
    let n = 12;
    let mut tn = loaded_backend(&crate::circuits::cz_chain_circuit(n, 4, 42));
    let before = planner_calls();
    tn.sample_native(8, 42).unwrap();
    assert_eq!(planner_calls() - before, n);
}

#[test]
fn test_sample_basis_states_repeats_from_the_seed() {
    let circuit = crate::circuits::cz_chain_circuit(6, 3, 42);
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(6, 0).unwrap();
    for inst in &circuit.instructions {
        tn.apply(inst).unwrap();
    }

    let first = tn.sample_basis_states(64, 42).unwrap();
    let second = tn.sample_basis_states(64, 42).unwrap();
    let other = tn.sample_basis_states(64, 43).unwrap();

    let bits = |s: &BasisSamples| -> Vec<bool> {
        (0..64)
            .flat_map(|shot| (0..6).map(move |q| (shot, q)))
            .map(|(shot, q)| s.bit(shot, q))
            .collect()
    };
    assert_eq!(bits(&first), bits(&second));
    assert_ne!(bits(&first), bits(&other));
}

// Two entangled components of unequal size, so join_disjoint merges tensors
// rather than bare scalars and its smallest-first order is observable.
#[test]
fn test_scalar_expectation_unequal_disjoint_components() {
    let mut circuit = Circuit::new(10, 0);
    for &q in &[0usize, 1, 2, 3, 4] {
        circuit.add_gate(Gate::Ry(0.3 + q as f64 * 0.1), &[q]);
    }
    for &(a, b) in &[(0usize, 1usize), (1, 2), (2, 3), (3, 4)] {
        circuit.add_gate(Gate::Cx, &[a, b]);
    }
    circuit.add_gate(Gate::H, &[7]);
    circuit.add_gate(Gate::Cx, &[7, 8]);
    let terms = [PauliTerm::z(0), PauliTerm::x(4), PauliTerm::z(7)];

    let expected = crate::sim::run_expectation_values(&circuit, &[terms.to_vec()], 42).unwrap()[0];
    let actual =
        expectation_zero_state(&circuit, &terms, &ResolvedLimits::process_default()).unwrap();
    assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
}
