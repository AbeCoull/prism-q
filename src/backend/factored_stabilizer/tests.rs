use crate::backend::Backend;
use crate::backend::factored_stabilizer::FactoredStabilizerBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::circuit::Circuit;
use crate::gates::Gate;

fn run_both(circuit: &Circuit, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut mono = StabilizerBackend::new(seed);
    mono.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for inst in &circuit.instructions {
        mono.apply(inst).unwrap();
    }
    let mono_probs = mono.probabilities().unwrap();

    let mut fact = FactoredStabilizerBackend::new(seed);
    fact.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for inst in &circuit.instructions {
        fact.apply(inst).unwrap();
    }
    let fact_probs = fact.probabilities().unwrap();

    (mono_probs, fact_probs)
}

fn assert_probs_eq(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "probability vector length mismatch");
    for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (pa - pb).abs() < tol,
            "prob[{}] differs: mono={} fact={}",
            i,
            pa,
            pb
        );
    }
}

#[test]
fn single_qubit_h() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn bell_state() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn ghz_3() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn ghz_8() {
    let n = 8;
    let circ = crate::circuits::ghz_circuit(n);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn independent_bell_pairs() {
    let circ = crate::circuits::independent_bell_pairs(4);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn clifford_heavy_small() {
    let circ = crate::circuits::clifford_heavy_circuit(6, 5, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn clifford_heavy_10q() {
    let circ = crate::circuits::clifford_heavy_circuit(10, 10, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn local_blocks_4x4() {
    let circ = crate::circuits::local_clifford_blocks(4, 4, 5, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn local_blocks_5x3() {
    let circ = crate::circuits::local_clifford_blocks(5, 3, 8, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn all_clifford_gates() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_gate(Gate::Cz, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn measurement_bell() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);

    let mut mono = StabilizerBackend::new(42);
    mono.init(2, 2).unwrap();
    for inst in &c.instructions {
        mono.apply(inst).unwrap();
    }

    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(2, 2).unwrap();
    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    assert_eq!(mono.classical_results(), fact.classical_results());
}

#[test]
fn measurement_ghz_all() {
    for seed in [42, 123, 999, 7777] {
        let n = 5;
        let mut c = Circuit::new(n, n);
        c.add_gate(Gate::H, &[0]);
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..n {
            c.add_measure(q, q);
        }

        let mut mono = StabilizerBackend::new(seed);
        mono.init(n, n).unwrap();
        for inst in &c.instructions {
            mono.apply(inst).unwrap();
        }

        let mut fact = FactoredStabilizerBackend::new(seed);
        fact.init(n, n).unwrap();
        for inst in &c.instructions {
            fact.apply(inst).unwrap();
        }

        assert_eq!(
            mono.classical_results(),
            fact.classical_results(),
            "seed={}",
            seed
        );
    }
}

#[test]
fn split_after_bell_measure() {
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(4, 2).unwrap();

    let mut c = Circuit::new(4, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_measure(0, 0);
    c.add_measure(2, 1);

    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    let active_count = fact.subs.iter().filter(|s| s.is_some()).count();
    assert!(
        active_count >= 2,
        "expected split after measurement, got {} active sub-tableaux",
        active_count
    );
}

// One merged cluster, not two that were never merged: measuring either half of
// a Bell pair leaves a product state, so the cluster must factor back apart.
#[test]
fn split_after_measuring_a_merged_cluster() {
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(2, 1).unwrap();

    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);

    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    let active_count = fact.subs.iter().filter(|s| s.is_some()).count();
    assert_eq!(
        active_count, 2,
        "a measured Bell pair is a product state and must split, got {} active sub-tableaux",
        active_count
    );
}

// A collapsed GHZ chain factors into one cluster per qubit, which exceeds
// the former 64-component ceiling in split detection.
#[test]
fn split_into_more_than_64_components() {
    let n = 100;
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(n, 1).unwrap();

    let mut c = Circuit::new(n, 1);
    c.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_measure(0, 0);

    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    let active_count = fact.subs.iter().filter(|s| s.is_some()).count();
    assert_eq!(
        active_count, n,
        "a measured GHZ chain is a product state and must split per qubit, got {} active \
         sub-tableaux",
        active_count
    );
}

// The deterministic measurement branch reads destabilizer bits, so a
// measurement on a cluster whose gates ran after the last materialization
// must rebuild them first. State is |11>, so measuring q1 must read 1; stale
// destabilizers from the gate-only window read 0.
#[test]
fn deterministic_measure_after_gates_rebuilds_destabilizers() {
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(2, 1).unwrap();

    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(1, 0);

    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    assert!(
        fact.classical_results()[0],
        "measuring q1 of |11> must read 1"
    );
}

// A merge drops the destabilizer rows of both sides, so a deterministic
// measurement after a cross-cluster gate must rebuild them for the merged
// tableau.
#[test]
fn deterministic_measure_after_merge() {
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(2, 3).unwrap();

    let mut c = Circuit::new(2, 3);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(1, 2);

    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    let bits = fact.classical_results();
    assert!(bits[0], "measuring q0 of |1> must read 1");
    assert!(!bits[1], "measuring q1 of |0> must read 0");
    assert!(bits[2], "after CX(0,1) on |10>, measuring q1 must read 1");
}

#[test]
fn product_state_stays_factored() {
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(4, 0).unwrap();

    let c = Circuit::new(4, 0);
    let mut c = c;
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(Gate::Y, &[3]);

    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    let active_count = fact.subs.iter().filter(|s| s.is_some()).count();
    assert_eq!(active_count, 4, "1q gates should not trigger merges");
}

#[test]
fn merge_then_probabilities() {
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(4, 0).unwrap();

    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);

    for inst in &c.instructions {
        fact.apply(inst).unwrap();
    }

    let active_count = fact.subs.iter().filter(|s| s.is_some()).count();
    assert_eq!(active_count, 3, "CX(0,1) merges q0,q1; q2,q3 stay separate");

    let probs = fact.probabilities().unwrap();
    assert_eq!(probs.len(), 16);

    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn clifford_random_pairs_12q() {
    let circ = crate::circuits::clifford_random_pairs(12, 5, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_cz_only() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_swap_only() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_single_block_3q() {
    let circ = crate::circuits::local_clifford_blocks(1, 3, 8, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_two_blocks_3q() {
    let circ = crate::circuits::local_clifford_blocks(2, 3, 8, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_three_blocks_3q() {
    let circ = crate::circuits::local_clifford_blocks(3, 3, 8, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_four_blocks_3q() {
    let circ = crate::circuits::local_clifford_blocks(4, 3, 8, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_five_blocks_3q() {
    let circ = crate::circuits::local_clifford_blocks(5, 3, 8, 0xDEAD_BEEF);
    let (mono, fact) = run_both(&circ, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_cz_swap_sequence() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_gate(Gate::Cz, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_merge_via_cx() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_merge_via_swap() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_1q_then_merge() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_full_1q_merge_cz_swap() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_full_1q_merge_all_cx() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_post_merge_h() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_post_merge_h_cx() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::Cx, &[0, 3]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_post_merge_h_cx_cz() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_gate(Gate::Cz, &[1, 2]);
    let (mono, fact) = run_both(&c, 42);
    assert_probs_eq(&mono, &fact, 1e-12);
}

#[test]
fn diag_verify_via_statevector() {
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[1]);
    c.add_gate(Gate::Sdg, &[2]);
    c.add_gate(Gate::X, &[3]);
    c.add_gate(Gate::Y, &[0]);
    c.add_gate(Gate::Z, &[1]);
    c.add_gate(Gate::SX, &[2]);
    c.add_gate(Gate::SXdg, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    c.add_gate(Gate::Swap, &[1, 2]);

    let mut sv = StatevectorBackend::new(42);
    sv.init(4, 0).unwrap();
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(4, 0).unwrap();

    for (step, inst) in c.instructions.iter().enumerate() {
        sv.apply(inst).unwrap();
        fact.apply(inst).unwrap();

        let active_count = fact.subs.iter().filter(|s| s.is_some()).count();
        let total_qubits: usize = fact
            .subs
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|s| s.n)
            .sum();
        if active_count == 1 && total_qubits == 4 {
            let sv_vec = sv.export_statevector().unwrap();
            let fact_vec = fact.export_statevector().unwrap();
            let mut differs = false;
            for i in 0..16 {
                if (sv_vec[i].re - fact_vec[i].re).abs() > 1e-10
                    || (sv_vec[i].im - fact_vec[i].im).abs() > 1e-10
                {
                    differs = true;
                    break;
                }
            }
            if differs {
                eprintln!("STATEVECTOR DIVERGES at step {} ({:?}):", step, inst);
                for i in 0..16 {
                    let s = sv_vec[i];
                    let f = fact_vec[i];
                    if s.norm() > 1e-12 || f.norm() > 1e-12 {
                        let diff = if (s.re - f.re).abs() > 1e-10 || (s.im - f.im).abs() > 1e-10 {
                            " DIFF"
                        } else {
                            ""
                        };
                        eprintln!(
                            "  |{:04b}⟩ sv={:.4}+{:.4}i  fact={:.4}+{:.4}i{}",
                            i, s.re, s.im, f.re, f.im, diff
                        );
                    }
                }
                panic!("Diverged at step {}", step);
            } else {
                eprintln!("Step {} ({:?}): statevectors match", step, inst);
            }
        } else {
            eprintln!(
                "Step {} ({:?}): {} subs, skipping sv compare",
                step, inst, active_count
            );
        }
    }
}

#[test]
fn non_clifford_rejected() {
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(2, 0).unwrap();
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::T, &[0]);
    let result = fact.apply(&c.instructions[0]);
    assert!(result.is_err());
}

#[test]
fn par_path_128q_all_clifford_gates() {
    let n = 130;
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::S, &[q]);
        c.add_gate(Gate::Sdg, &[q]);
        c.add_gate(Gate::X, &[q]);
        c.add_gate(Gate::Y, &[q]);
        c.add_gate(Gate::Z, &[q]);
        c.add_gate(Gate::SX, &[q]);
        c.add_gate(Gate::SXdg, &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::Cz, &[0, n - 1]);
    c.add_gate(Gate::Swap, &[1, n - 2]);
    let mut b = FactoredStabilizerBackend::new(42);
    crate::sim::run_on(&mut b, &c).unwrap();
}

// Pins the interleaved-measure bench shape: a linear cluster state on qubits
// 1..n with qubit 0 re-attached as a leaf each round. The leaf measurement must
// split off only the leaf, leaving the chain one entangled cluster, so every
// round's measurement lands on the full cluster with gates applied since the
// last one.
#[test]
fn interleaved_leaf_measure_keeps_chain_entangled() {
    let n = 12;
    let rounds = 5;
    let mut fact = FactoredStabilizerBackend::new(42);
    fact.init(n, rounds).unwrap();

    let mut init = Circuit::new(n, rounds);
    for q in 1..n {
        init.add_gate(Gate::H, &[q]);
    }
    for q in 1..n - 1 {
        init.add_gate(Gate::Cz, &[q, q + 1]);
    }
    for inst in &init.instructions {
        fact.apply(inst).unwrap();
    }

    for round in 0..rounds {
        let mut layer = Circuit::new(n, rounds);
        for q in 0..n {
            layer.add_gate(Gate::S, &[q]);
        }
        layer.add_gate(Gate::H, &[0]);
        layer.add_gate(Gate::Cz, &[0, 1]);
        for inst in &layer.instructions {
            fact.apply(inst).unwrap();
        }

        let merged_count = fact.subs.iter().filter(|s| s.is_some()).count();
        assert_eq!(
            merged_count, 1,
            "round {}: expected one merged cluster before the measurement, got {}",
            round, merged_count
        );

        let mut measure = Circuit::new(n, rounds);
        measure.add_measure(0, round);
        fact.apply(&measure.instructions[0]).unwrap();

        let active_count = fact.subs.iter().filter(|s| s.is_some()).count();
        assert_eq!(
            active_count, 2,
            "round {}: the measured leaf must split off alone, got {} active sub-tableaux",
            round, active_count
        );
        let chain = fact.qubit_to_sub[1];
        assert_eq!(
            fact.subs[chain].as_ref().unwrap().n,
            n - 1,
            "round {}: the chain must stay one cluster of {} qubits",
            round,
            n - 1
        );
    }
}
