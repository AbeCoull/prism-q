use std::cmp::Reverse;

use super::expectation::{FUSED_PLAN_TOTAL, ScalarExpectationNetwork};
use super::plan::{
    PLANNER_CALLS, RESTART_PEAK_THRESHOLD, contract_meta, contraction_result_size,
    plan_with_restarts,
};
use super::slice::{
    SLICE_SEARCHES, cached_plan, choose_slices, contract_slices, contract_within, kept_rank,
    pin_axes, replay_plan, split_bond,
};
use super::*;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Circuit;
use crate::gates::{Gate, MultiFusedData};
use crate::limits::LimitSource;
use crate::test_support::assert_probs_close;

const EPS: f64 = 1e-10;

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
        EPS,
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
        EPS,
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

    assert_probs_close(&tn_probs, &sv_probs, EPS);
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

    assert_probs_close(
        &tn.probabilities().unwrap(),
        &sv.probabilities().unwrap(),
        EPS,
    );
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

// The build before gates were screened by the cone: every gate's ops, then the
// cone over ops, then idle tensors dropped from a network over every qubit.
fn unscreened_network(circuit: &Circuit, terms: &[PauliTerm]) -> ScalarExpectationNetwork {
    let n = circuit.num_qubits;
    let mut axes = vec![None; n];
    observable_axes(terms, &mut axes, "test").unwrap();
    let mut ops = Vec::new();
    for instruction in &circuit.instructions {
        if let Instruction::Gate { gate, targets } = instruction {
            for_each_gate_tensor(gate, targets, n, |op| {
                ops.push(op);
                Ok(())
            })
            .unwrap();
        }
    }
    let mut in_cone: Vec<bool> = axes.iter().map(Option::is_some).collect();
    let mut keep = vec![false; ops.len()];
    for (kept, op) in keep.iter_mut().zip(&ops).rev() {
        let qubits = op.qubits();
        if qubits.iter().any(|&q| in_cone[q]) {
            *kept = true;
            for q in qubits {
                in_cone[q] = true;
            }
        }
    }
    let mut network = ScalarExpectationNetwork::new(n, &vec![true; n]);
    let mut idle_legs = vec![false; network.next_leg];
    for q in (0..n).filter(|&q| !in_cone[q]) {
        idle_legs[network.ket_legs[q]] = true;
        idle_legs[network.bra_legs[q]] = true;
    }
    for (op, kept) in ops.into_iter().zip(keep) {
        if kept {
            network.append_op(op);
        }
    }
    for (qubit, axis) in axes.into_iter().enumerate() {
        let tensor = pauli_tensor(axis, network.bra_legs[qubit], network.ket_legs[qubit]);
        network.tensors.push(tensor);
    }
    network
        .tensors
        .retain(|t| !t.legs.iter().all(|&leg| idle_legs.get(leg) == Some(&true)));
    network
}

// Screening gates by the cone before building them keeps the same tensors in
// the same order, with legs renumbered in order, so the greedy plan and the
// value are unchanged; the fused QFT carries BatchPhase, whose targets omit
// the phase qubits.
#[test]
fn test_scalar_network_screened_by_the_cone_matches_the_unscreened_build() {
    let mut prepared_qft = crate::circuits::hardware_efficient_ansatz(16, 1, 42);
    let qft = crate::circuits::qft_circuit(16);
    prepared_qft.instructions.extend(
        crate::circuit::expand_qft_blocks(&qft)
            .instructions
            .iter()
            .cloned(),
    );
    let fused_qft = crate::circuit::fusion::fuse_circuit(&prepared_qft, true).into_owned();
    assert!(fused_qft.instructions.iter().any(|i| matches!(
        i,
        Instruction::Gate {
            gate: Gate::BatchPhase(_),
            ..
        }
    )));
    let cases: Vec<(Circuit, Vec<PauliTerm>)> = vec![
        (
            crate::circuits::brickwork_circuit(14, 4, 42),
            vec![PauliTerm::z(0)],
        ),
        (
            crate::circuits::hardware_efficient_ansatz(12, 3, 42),
            vec![PauliTerm::z(4), PauliTerm::x(5)],
        ),
        (
            crate::circuits::quantum_volume_circuit(10, 3, 42),
            vec![PauliTerm::y(2)],
        ),
        (fused_qft.clone(), vec![PauliTerm::z(3)]),
        (fused_qft, vec![PauliTerm::x(9), PauliTerm::z(15)]),
    ];
    for (circuit, terms) in cases {
        let screened = scalar_network(&circuit, &terms);
        let reference = unscreened_network(&circuit, &terms);
        assert_eq!(screened.tensors.len(), reference.tensors.len());
        let mut leg_map = std::collections::BTreeMap::new();
        for (a, b) in screened.tensors.iter().zip(&reference.tensors) {
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.data, b.data);
            for (&la, &lb) in a.legs.iter().zip(&b.legs) {
                assert_eq!(*leg_map.entry(lb).or_insert(la), la);
            }
        }
        let renumbered: Vec<_> = leg_map.values().collect();
        assert!(renumbered.windows(2).all(|w| w[0] < w[1]));

        let plan = |network: &ScalarExpectationNetwork| {
            let slots = network.tensors.iter().map(TensorMeta::of).collect();
            plan_pairs(slots, None, usize::MAX, usize::MAX).unwrap()
        };
        let (a, b) = (plan(&screened), plan(&reference));
        assert_eq!((a.pairs, a.peak, a.total), (b.pairs, b.peak, b.total));

        let mut sv = StatevectorBackend::new(42);
        sv.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        for instruction in &circuit.instructions {
            sv.apply(instruction).unwrap();
        }
        let expected = sv.pauli_expectations(std::slice::from_ref(&terms)).unwrap()[0];
        let actual =
            expectation_zero_state(&circuit, &terms, &ResolvedLimits::process_default()).unwrap();
        assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
    }
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
    let slots: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();

    let greedy = plan_pairs(slots, None, usize::MAX, usize::MAX).unwrap();
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
fn test_cached_slice_choice_matches_the_uncached_search() {
    let circuit = crate::circuits::hardware_efficient_ansatz(12, 4, 42);
    let terms = [PauliTerm::z(0), PauliTerm::z(6)];
    let network = scalar_network(&circuit, &terms);
    let plan = plan_with_restarts(&network.tensors);
    let metas: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();
    let limits = ContractionLimits {
        peak_cap: 1 << 8,
        peak_source: LimitSource::Request,
        slice_budget: 1 << 10,
        tolerance: None,
    };
    assert!(plan.peak > limits.peak_cap, "the fixture never slices");
    let uncached = choose_slices(&metas, &plan, limits).expect("the budget reaches the cap");

    let mut tensors = network.tensors.clone();
    let direct = contract_within(&mut tensors, &plan, limits, "tn", "test").unwrap();

    let mut slot = None;
    for expected_searches in [1, 0] {
        let before = slice_searches();
        let mut tensors = network.tensors.clone();
        let cached = contract_cached(&mut tensors, &mut slot, limits, "tn", "test").unwrap();
        assert_eq!(slot.as_ref().unwrap().cached_slice(), Some(Some(&uncached)));
        assert_eq!(slice_searches() - before, expected_searches);
        crate::test_support::assert_amps_close(&cached.data, &direct.data, 1e-12);
    }
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
        let slots: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let plan = plan_pairs(slots, Some((&mut rng, 1.0)), usize::MAX, usize::MAX).unwrap();

        let result = replay_plan(&mut network.tensors, &plan);
        assert_eq!(result.data.len(), 1);
        assert!(
            (result.data[0].re - expected).abs() < EPS,
            "seed {seed}: {} vs {expected}",
            result.data[0].re
        );
    }
}

// Extents of 1 to 3, a few open legs, and endpoints drawn with repeats, so the
// exact queue meets sizes off the powers of two and pairs sharing several legs.
fn random_network(seed: u64) -> Vec<TensorMeta> {
    use rand::RngExt;
    let tensors = 20;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut metas = vec![
        TensorMeta {
            shape: SmallVec::new(),
            legs: SmallVec::new(),
        };
        tensors
    ];
    for leg in 0..36 {
        let extent = rng.random_range(1..=3);
        let a = rng.random_range(0..tensors);
        let b = (a + rng.random_range(1..tensors)) % tensors;
        let ends: &[usize] = if leg % 9 == 0 { &[a] } else { &[a, b] };
        for &end in ends {
            metas[end].shape.push(extent);
            metas[end].legs.push(leg);
        }
    }
    metas
}

// At temperature zero the noisy key is the result size itself, so the heap arm
// pops in the order the exact pass's size stacks have to reproduce.
#[test]
fn test_exact_plans_match_the_heap_order() {
    let mut networks: Vec<Vec<TensorMeta>> = (0..4).map(random_network).collect();
    for (n, layers) in [(20, 2), (50, 2), (30, 7)] {
        let circuit = crate::circuits::hardware_efficient_ansatz(n, layers, 42);
        let terms = [PauliTerm::z(0), PauliTerm::z(n / 2)];
        let network = scalar_network(&circuit, &terms);
        networks.push(network.tensors.iter().map(TensorMeta::of).collect());
    }
    let chain = loaded_backend(&crate::circuits::cz_chain_circuit(40, 8, 42));
    let doubled = chain.double_for_partial_trace(&[20]);
    networks.push(doubled.iter().map(TensorMeta::of).collect());

    for metas in &networks {
        let exact = plan_pairs(metas.clone(), None, usize::MAX, usize::MAX).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let heap =
            plan_pairs(metas.clone(), Some((&mut rng, 0.0)), usize::MAX, usize::MAX).unwrap();
        assert_eq!(exact.pairs, heap.pairs);
        assert_eq!((exact.peak, exact.total), (heap.peak, heap.total));
    }
}

// Every live pair sharing a leg sits in the queue under a key fixed at push
// time, so the exact pass must pick what a rescan of all live pairs picks:
// smallest result, then the higher slot descending, then the lower.
#[test]
fn test_exact_plans_match_a_rescanning_greedy() {
    let mut networks: Vec<Vec<TensorMeta>> = (0..40).map(random_network).collect();
    for n in [20, 50] {
        let circuit = crate::circuits::hardware_efficient_ansatz(n, 2, 0xDEAD_BEEF);
        let terms = [PauliTerm::z(0), PauliTerm::z(n / 2)];
        let network = scalar_network(&circuit, &terms);
        networks.push(network.tensors.iter().map(TensorMeta::of).collect());
    }
    for depth in [4, 8] {
        let chain = loaded_backend(&crate::circuits::cz_chain_circuit(40, depth, 0xDEAD_BEEF));
        let doubled = chain.double_for_partial_trace(&[20]);
        networks.push(doubled.iter().map(TensorMeta::of).collect());
    }

    for metas in &networks {
        let mut slots: Vec<Option<TensorMeta>> = metas.iter().cloned().map(Some).collect();
        let mut pairs = Vec::new();
        loop {
            let mut best: Option<(usize, Reverse<usize>, Reverse<usize>)> = None;
            for j in 0..slots.len() {
                for i in 0..j {
                    let (Some(a), Some(b)) = (&slots[i], &slots[j]) else {
                        continue;
                    };
                    if a.legs.iter().any(|leg| b.legs.contains(leg)) {
                        let key = (contraction_result_size(a, b), Reverse(j), Reverse(i));
                        best = Some(best.map_or(key, |held| held.min(key)));
                    }
                }
            }
            let Some((_, Reverse(j), Reverse(i))) = best else {
                break;
            };
            let a = slots[i].take().unwrap();
            let b = slots[j].take().unwrap();
            slots.push(Some(contract_meta(&a, &b)));
            pairs.push((i, j));
        }
        let plan = plan_pairs(metas.clone(), None, usize::MAX, usize::MAX).unwrap();
        assert_eq!(plan.pairs, pairs);
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
        assert_probs_close(
            &tn.probabilities().unwrap(),
            &sv.probabilities().unwrap(),
            EPS,
        );
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

    assert_probs_close(&tn.probabilities().unwrap(), &probs_before, EPS);
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

fn slice_searches() -> usize {
    SLICE_SEARCHES.with(|calls| calls.get())
}

// Sample hardware_efficient_ansatz(8, 3) through the sweep at a 2^7 peak cap,
// once searching on every contraction and once through the cache, and count
// the slice searches of each run.
fn sample_capped_with_and_without_cache(tolerance: Option<f64>, shots: usize) -> (usize, usize) {
    let n = 8;
    let limits = ResolvedLimits::resolve(Some(&crate::Limits::new().max_tn_peak_qubits(7)));
    let mut tn = TensorNetworkBackend::with_limits(42, tolerance, &limits);
    tn.init(n, 0).unwrap();
    for inst in &crate::circuits::hardware_efficient_ansatz(n, 3, 42).instructions {
        tn.apply(inst).unwrap();
    }

    let before = slice_searches();
    let uncached = tn.sample_sweep(shots, 42, None).unwrap();
    let uncached_searches = slice_searches() - before;
    assert!(uncached_searches >= shots, "the fixture never sliced");

    let before = slice_searches();
    let cached = tn.sample_native(shots, 42).unwrap();
    assert_eq!(cached.words, uncached.words);
    (slice_searches() - before, uncached_searches)
}

#[test]
fn test_slice_cache_searches_each_position_once() {
    let shots = 4;
    let (cached, uncached) = sample_capped_with_and_without_cache(None, shots);
    assert_eq!(cached * shots, uncached);
}

// Truncation picks its kept rank from the data, so the network it leaves to
// slicing can change between shots; the cache only has to stay transparent
// and reuse a choice where the shape repeats.
#[test]
fn test_slice_cache_matches_uncached_truncated_sweep() {
    let (cached, uncached) = sample_capped_with_and_without_cache(Some(1e-3), 4);
    assert!(cached < uncached, "{cached} searches against {uncached}");
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

fn pauli_2x2(letter: usize) -> [[Complex64; 2]; 2] {
    let (o, l, i) = (
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    );
    match letter {
        0 => [[l, o], [o, l]],
        1 => [[o, l], [l, o]],
        2 => [[o, -i], [i, o]],
        _ => [[l, o], [o, -l]],
    }
}

// Tr(rho P) for every Pauli string on the subsystem, against the sandwich
// contraction, at a width where no dense kernel is consulted: a swapped bit
// order or a conjugated ket and bra moves a Y string's sign.
#[test]
fn rdm_traces_against_pauli_strings_match_pauli_expectations() {
    let mut tn = loaded_backend(&crate::circuits::cz_chain_circuit(24, 4, 42));
    for subsystem in [vec![11usize, 12], vec![20, 3], vec![2, 13, 21]] {
        let k = subsystem.len();
        let dim = 1usize << k;
        let rho = tn.reduced_density_matrix(&subsystem).unwrap();
        for word in 1..1usize << (2 * k) {
            let letters: Vec<usize> = (0..k).map(|i| (word >> (2 * i)) & 3).collect();
            let terms: Vec<PauliTerm> = subsystem
                .iter()
                .zip(&letters)
                .filter_map(|(&q, &letter)| match letter {
                    1 => Some(PauliTerm::x(q)),
                    2 => Some(PauliTerm::y(q)),
                    3 => Some(PauliTerm::z(q)),
                    _ => None,
                })
                .collect();
            let mut trace = Complex64::new(0.0, 0.0);
            for t in 0..dim {
                for tp in 0..dim {
                    let p = (0..k).fold(Complex64::new(1.0, 0.0), |acc, i| {
                        acc * pauli_2x2(letters[i])[(tp >> i) & 1][(t >> i) & 1]
                    });
                    trace += rho[t * dim + tp] * p;
                }
            }
            let expected = tn.pauli_expectations(&[terms]).unwrap()[0];
            assert!(
                (trace.re - expected).abs() < 1e-10 && trace.im.abs() < 1e-10,
                "{subsystem:?} word {word}: {trace} against {expected}"
            );
        }
    }
}

#[test]
fn one_qubit_general_matches_marginal_1q() {
    let mut tn = loaded_backend(&crate::circuits::random_circuit(8, 4, 42));
    for q in 0..8 {
        let one = tn.reduced_density_matrix_1q(q).unwrap();
        let rho = tn.reduced_density_matrix(&[q]).unwrap();
        for (a, b) in rho.iter().zip(one.iter().flatten()) {
            assert!((a - b).norm() < 1e-12, "qubit {q}: {a} against {b}");
        }
    }
}

// A Bell pair on (0, 1), Ry(theta) on 3, nothing on 2 and 4: the kept qubits
// sit in disconnected components, so the result is an outer product of the
// pair's projector onto |00> + |11>, the pure Ry state, and |0><0|.
#[test]
fn rdm_with_idle_and_disconnected_qubits() {
    let theta = 0.7f64;
    let mut circuit = Circuit::new(5, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Ry(theta), &[3]);
    let mut tn = loaded_backend(&circuit);
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let amp = |q: usize, bit: usize| match (q, bit) {
        (3, 0) => c,
        (3, _) => s,
        (_, 0) => 1.0,
        _ => 0.0,
    };
    for subsystem in [vec![2usize, 0], vec![4, 3], vec![1, 0, 3]] {
        let k = subsystem.len();
        let dim = 1usize << k;
        let rho = tn.reduced_density_matrix(&subsystem).unwrap();
        for t in 0..dim {
            for tp in 0..dim {
                let bit = |i: usize, x: usize| (x >> i) & 1;
                let mut expected = 1.0;
                let mut pair = [None, None];
                for (i, &q) in subsystem.iter().enumerate() {
                    if q < 2 {
                        pair[q] = Some((bit(i, t), bit(i, tp)));
                    } else {
                        expected *= amp(q, bit(i, t)) * amp(q, bit(i, tp));
                    }
                }
                expected *= match pair {
                    [Some((a, ap)), Some((b, bp))] if a == b && ap == bp => 0.5,
                    [Some(_), Some(_)] => 0.0,
                    [Some((a, ap)), None] | [None, Some((a, ap))] if a == ap => 0.5,
                    [Some(_), None] | [None, Some(_)] => 0.0,
                    [None, None] => 1.0,
                };
                let entry = rho[t * dim + tp];
                assert!(
                    (entry - Complex64::new(expected, 0.0)).norm() < 1e-12,
                    "{subsystem:?} ({t}, {tp}): {entry} against {expected}"
                );
            }
        }
    }
}

#[test]
fn test_double_cone_prunes_inactive_unitary_components() {
    let mut circuit = Circuit::new(10, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    for q in 2..10 {
        circuit.add_gate(Gate::Ry(0.3 + q as f64 * 0.1), &[q]);
    }
    for q in 2..9 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }

    let tn = loaded_backend(&circuit);
    // Component {0, 1} has 2 kets + H(0) + Cx(0, 1) = 4 tensors (8 doubled).
    let partial = tn.double_for_partial_trace(&[0]);
    assert_eq!(partial.len(), 8);
    assert!(partial.len() < tn.tensors.len() * 2);

    let mut axes = [None; 10];
    axes[1] = Some(PauliAxis::Z);
    let sandwich = tn.double_cone(|q| axes[q].is_some());
    assert_eq!(sandwich.len(), 8);
}

#[test]
fn test_pauli_expectations_skips_norm_when_unitary_and_handles_identity_and_empty() {
    let circuit = crate::circuits::cz_chain_circuit(6, 3, 42);
    let tn = loaded_backend(&circuit);

    let before = planner_calls();
    let exps = tn
        .pauli_expectations(&[vec![PauliTerm::z(0)], vec![PauliTerm::x(3)]])
        .unwrap();
    assert_eq!(planner_calls() - before, 2);

    let mut sv = StatevectorBackend::new(42);
    sv.init(6, 0).unwrap();
    for inst in &circuit.instructions {
        sv.apply(inst).unwrap();
    }
    let sv_exps = sv
        .pauli_expectations(&[vec![PauliTerm::z(0)], vec![PauliTerm::x(3)]])
        .unwrap();
    for (got, want) in exps.iter().zip(&sv_exps) {
        assert!((got - want).abs() < EPS, "{got} vs {want}");
    }

    let before = planner_calls();
    let id_exp = tn.pauli_expectations(&[Vec::new()]).unwrap();
    assert_eq!(planner_calls() - before, 0);
    assert!((id_exp[0] - 1.0).abs() < EPS);

    let mut empty = TensorNetworkBackend::new(42);
    empty.init(0, 0).unwrap();
    let before = planner_calls();
    let empty_exp = empty.pauli_expectations(&[Vec::new()]).unwrap();
    assert_eq!(planner_calls() - before, 0);
    assert!((empty_exp[0] - 1.0).abs() < EPS);
}

#[test]
fn test_non_unitary_reset_after_entangler_is_never_pruned() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_reset(0);

    for seed in [42u64, 7, 99] {
        let mut sv = StatevectorBackend::new(seed);
        sv.init(2, 0).unwrap();
        let mut tn = TensorNetworkBackend::new(seed);
        tn.init(2, 0).unwrap();
        for inst in &circuit.instructions {
            sv.apply(inst).unwrap();
            tn.apply(inst).unwrap();
        }
        assert!(!tn.is_unitary[0]);
        assert!(tn.is_unitary[1]);

        let sv_rdm = sv.reduced_density_matrix_1q(1).unwrap();
        let tn_rdm = tn.reduced_density_matrix_1q(1).unwrap();
        for (row_a, row_b) in tn_rdm.iter().zip(&sv_rdm) {
            for (a, b) in row_a.iter().zip(row_b) {
                assert!((a - b).norm() < EPS, "seed {seed}: {a} vs {b}");
            }
        }

        let obs = [vec![PauliTerm::x(1)], vec![PauliTerm::z(1)]];
        let sv_exp = sv.pauli_expectations(&obs).unwrap();
        let tn_exp = tn.pauli_expectations(&obs).unwrap();
        for (a, b) in tn_exp.iter().zip(&sv_exp) {
            assert!((a - b).abs() < EPS, "seed {seed}: {a} vs {b}");
        }
    }
}

#[test]
fn test_downstream_unitary_pruned_after_measure_closes_internal_leg() {
    let mut circuit = Circuit::new(6, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::H, &[1]);
    circuit.add_gate(Gate::Rz(0.5), &[1]);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_gate(Gate::Ry(0.8), &[4]);
    circuit.add_gate(Gate::Cx, &[4, 5]);

    for seed in [42u64, 17] {
        let mut sv = StatevectorBackend::new(seed);
        sv.init(6, 1).unwrap();
        let mut tn = TensorNetworkBackend::new(seed);
        tn.init(6, 1).unwrap();
        for inst in &circuit.instructions {
            sv.apply(inst).unwrap();
            tn.apply(inst).unwrap();
        }
        assert!(!tn.is_unitary[0]);
        assert!(tn.is_unitary[1]);

        // Only |0>_0, |0>_1, H(0), Cx(0, 1)+Measure(0) are in the doubled cone;
        // H(1), Rz(0.5, 1), and qubits 2..6 are pruned.
        let doubled = tn.double_for_partial_trace(&[0]);
        assert_eq!(doubled.len(), 8);

        let mut leg_counts = std::collections::HashMap::<LegId, usize>::new();
        for tensor in &doubled {
            for &leg in &tensor.legs {
                *leg_counts.entry(leg).or_insert(0) += 1;
            }
        }
        let open_ket = tn.output_legs[0];
        let open_bra = open_ket + tn.next_leg;
        for (leg, count) in leg_counts {
            let expected_count = if leg == open_ket || leg == open_bra {
                1
            } else {
                2
            };
            assert_eq!(count, expected_count, "leg {leg} count");
        }

        let obs = [vec![PauliTerm::z(0)], vec![PauliTerm::z(1)]];
        let sv_exp = sv.pauli_expectations(&obs).unwrap();
        let tn_exp = tn.pauli_expectations(&obs).unwrap();
        for (a, b) in tn_exp.iter().zip(&sv_exp) {
            assert!((a - b).abs() < EPS, "seed {seed}: {a} vs {b}");
        }
    }
}

#[test]
fn test_apply_1q_matrix_shrink_entangled_and_disconnected() {
    let proj_half = [
        [Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ];

    // Entangled: H(0) -> Cx(0, 1) -> shrink(0)
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
    tn.apply_1q_matrix(0, &proj_half).unwrap();
    assert!(!tn.is_unitary[0]);
    assert!(tn.is_unitary[1]);

    let exps = tn
        .pauli_expectations(&[vec![PauliTerm::z(0)], vec![PauliTerm::z(1)]])
        .unwrap();
    assert!((exps[0] - 1.0).abs() < EPS, "{}", exps[0]);
    assert!((exps[1] - 1.0).abs() < EPS, "{}", exps[1]);
    let rdm1 = tn.reduced_density_matrix(&[1]).unwrap();
    assert!((rdm1[0].re - 1.0).abs() < EPS);
    assert!(rdm1[3].re.abs() < EPS);

    // Disconnected: shrink(0), H(1), and unitary component (2, 3)
    let mut disc = TensorNetworkBackend::new(42);
    disc.init(4, 0).unwrap();
    disc.apply_1q_matrix(0, &proj_half).unwrap();
    disc.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![1],
    })
    .unwrap();
    disc.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![2],
    })
    .unwrap();
    disc.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: smallvec::smallvec![2, 3],
    })
    .unwrap();

    // Querying X(1) keeps non-unitary qubit 0 (2 tensors) and open qubit 1 (2 tensors),
    // pruning (2, 3), so 4 ket tensors double to 8.
    let mut axes = [None; 4];
    axes[1] = Some(PauliAxis::X);
    assert_eq!(disc.double_cone(|q| axes[q].is_some()).len(), 8);

    let disc_exps = disc
        .pauli_expectations(&[vec![PauliTerm::x(1)], vec![PauliTerm::z(1)]])
        .unwrap();
    assert!((disc_exps[0] - 1.0).abs() < EPS, "{}", disc_exps[0]);
    assert!(disc_exps[1].abs() < EPS, "{}", disc_exps[1]);
}

#[test]
fn test_non_unitary_gate_variants_track_unitarity() {
    use crate::gates::{BatchPhaseData, DiagEntry, DiagonalBatchData, McuData, Multi2qData};

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Non-unitary Gate::Fused2q (Kraus 2q style) on (0, 1), entangled with qubit 2.
    let mut tn = TensorNetworkBackend::new(42);
    tn.init(3, 0).unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::H,
        targets: smallvec::smallvec![1],
    })
    .unwrap();
    tn.apply(&Instruction::Gate {
        gate: Gate::Cx,
        targets: smallvec::smallvec![1, 2],
    })
    .unwrap();
    let kraus_2q = [
        [Complex64::new(0.6, 0.0), zero, zero, zero],
        [zero, Complex64::new(0.6, 0.0), zero, zero],
        [zero, zero, zero, zero],
        [zero, zero, zero, zero],
    ];
    tn.apply(&Instruction::Gate {
        gate: Gate::Fused2q(Box::new(kraus_2q)),
        targets: smallvec::smallvec![0, 1],
    })
    .unwrap();
    assert!(!tn.is_unitary[0]);
    assert!(!tn.is_unitary[1]);
    assert!(tn.is_unitary[2]);

    // After projecting qubit 0 to |0>, qubit 1 stays in |0>+|1> and qubit 2 is
    // maximally mixed via Cx(1, 2): <Z(0)> = 1, <Z(1)> = 0, <Z(2)> = 0, <Z(1)Z(2)> = 1.
    let exps = tn
        .pauli_expectations(&[
            vec![PauliTerm::z(0)],
            vec![PauliTerm::z(1)],
            vec![PauliTerm::z(2)],
            vec![PauliTerm::z(1), PauliTerm::z(2)],
        ])
        .unwrap();
    assert!((exps[0] - 1.0).abs() < EPS, "{}", exps[0]);
    assert!(exps[1].abs() < EPS, "{}", exps[1]);
    assert!(exps[2].abs() < EPS, "{}", exps[2]);
    assert!((exps[3] - 1.0).abs() < EPS, "{}", exps[3]);

    // Check remaining matrix/phase variants including NaN handling.
    let mut variants = TensorNetworkBackend::new(42);
    variants.init(11, 0).unwrap();
    let nan_mat_2 = [[Complex64::new(f64::NAN, 0.0), zero], [zero, one]];
    let bad_mat_2 = [[Complex64::new(0.5, 0.0), zero], [zero, one]];
    let mut bad_mat_4 = Gate::Cx.matrix_4x4();
    bad_mat_4[0][0] = Complex64::new(0.5, 0.0);

    // Unitary Fused & Cu keep unitarity true.
    variants
        .apply(&Instruction::Gate {
            gate: Gate::Fused(Box::new(Gate::H.matrix_2x2())),
            targets: smallvec::smallvec![10],
        })
        .unwrap();
    variants
        .apply(&Instruction::Gate {
            gate: Gate::Cu(Box::new(Gate::X.matrix_2x2())),
            targets: smallvec::smallvec![9, 10],
        })
        .unwrap();
    assert!(variants.is_unitary[9] && variants.is_unitary[10]);

    // Non-unitary / NaN payloads clear only their affected qubits.
    variants
        .apply(&Instruction::Gate {
            gate: Gate::Fused(Box::new(nan_mat_2)),
            targets: smallvec::smallvec![0],
        })
        .unwrap();
    variants
        .apply(&Instruction::Gate {
            gate: Gate::Cu(Box::new(bad_mat_2)),
            targets: smallvec::smallvec![1, 2],
        })
        .unwrap();
    variants
        .apply(&Instruction::Gate {
            gate: Gate::Mcu(Box::new(McuData::new(bad_mat_2, 2))),
            targets: smallvec::smallvec![3, 4, 5],
        })
        .unwrap();
    variants
        .apply(&Instruction::Gate {
            gate: Gate::MultiFused(Box::new(MultiFusedData::new(vec![
                (6, Gate::H.matrix_2x2()),
                (7, bad_mat_2),
            ]))),
            targets: smallvec::smallvec![6, 7],
        })
        .unwrap();
    variants
        .apply(&Instruction::Gate {
            gate: Gate::Multi2q(Box::new(Multi2qData::new(vec![(8, 9, bad_mat_4)]))),
            targets: smallvec::smallvec![8, 9],
        })
        .unwrap();
    assert!(!variants.is_unitary[0]);
    assert!(!variants.is_unitary[1] && !variants.is_unitary[2]);
    assert!(!variants.is_unitary[3] && !variants.is_unitary[4] && !variants.is_unitary[5]);
    assert!(variants.is_unitary[6]);
    assert!(!variants.is_unitary[7]);
    assert!(!variants.is_unitary[8] && !variants.is_unitary[9]);
    assert!(variants.is_unitary[10]);

    let mut diag = TensorNetworkBackend::new(42);
    diag.init(5, 0).unwrap();
    diag.apply(&Instruction::Gate {
        gate: Gate::BatchPhase(Box::new(BatchPhaseData::new([
            (1, Complex64::new(0.0, 1.0)),
            (2, Complex64::new(2.0, 0.0)),
        ]))),
        targets: smallvec::smallvec![0],
    })
    .unwrap();
    assert!(!diag.is_unitary[0] && diag.is_unitary[1] && !diag.is_unitary[2]);

    diag.apply(&Instruction::Gate {
        gate: Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(vec![
            DiagEntry::Phase1q {
                qubit: 3,
                d0: one,
                d1: Complex64::new(0.0, -1.0),
            },
            DiagEntry::Parity2q {
                q0: 1,
                q1: 4,
                same: one,
                diff: Complex64::new(f64::NAN, 0.0),
            },
        ]))),
        targets: smallvec::smallvec![1, 3, 4],
    })
    .unwrap();
    assert!(diag.is_unitary[3]);
    assert!(!diag.is_unitary[1] && !diag.is_unitary[4]);
}

#[test]
fn test_sample_native_restores_is_unitary_across_shots() {
    let n = 8;
    let mut tn = loaded_backend(&crate::circuits::cz_chain_circuit(n, 3, 42));
    assert!(tn.is_unitary.iter().all(|&u| u));

    let before = planner_calls();
    let _ = tn.sample_native(4, 42).unwrap();
    assert_eq!(planner_calls() - before, n);
    assert!(tn.is_unitary.iter().all(|&u| u));
}
