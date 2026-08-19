//! Cross-backend correctness for `FactoredBackend`: SV cross checks and
//! fused-vs-unfused self-checks over the builder corpus at 8-12q, blocked
//! builders at 20q for the split-state path, monolithic builders at 16q.

mod common;

use common::{FACTORED_EPS, SEED, assert_backend_matches_sv, assert_fused_matches_unfused};
use prism_q::backend::factored::FactoredBackend;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::circuits;
use prism_q::gates::Gate;

fn check_sv_cross(label: &str, circuit: &Circuit) {
    let mut backend = FactoredBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, FACTORED_EPS, label);
}

fn check_fused_vs_unfused(label: &str, circuit: &Circuit) {
    assert_fused_matches_unfused(|| FactoredBackend::new(SEED), circuit, FACTORED_EPS, label);
}

fn has_multi_fused(circuit: &Circuit) -> bool {
    let fused = prism_q::circuit::fusion::fuse_circuit(circuit, true);
    fused.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate {
                gate: Gate::MultiFused(_),
                ..
            }
        )
    })
}

fn has_batch_phase(circuit: &Circuit) -> bool {
    let fused = prism_q::circuit::fusion::fuse_circuit(circuit, true);
    fused.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate {
                gate: Gate::BatchPhase(_),
                ..
            }
        )
    })
}

// ===== qft =====

#[test]
fn factored_qft_12q_sv() {
    check_sv_cross("qft 12q sv", &circuits::qft_circuit(12));
}

#[test]
fn factored_qft_12q_fused() {
    check_fused_vs_unfused("qft 12q fused", &circuits::qft_circuit(12));
}

#[test]
fn factored_qft_16q_sv() {
    check_sv_cross("qft 16q sv", &circuits::qft_circuit(16));
}

#[test]
fn factored_batch_phase_phase_sensitive_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::cphase(0.37), &[0, 1]);
    c.add_gate(Gate::cphase(-0.91), &[0, 2]);
    c.add_gate(Gate::H, &[0]);
    assert!(
        has_batch_phase(&c),
        "phase-sensitive 16q circuit should use BatchPhase fusion"
    );
    check_sv_cross("batch_phase phase-sensitive 16q sv", &c);
    check_fused_vs_unfused("batch_phase phase-sensitive 16q fused", &c);
}

// ===== random =====

#[test]
fn factored_random_12q_sv() {
    check_sv_cross("random 12q d10 sv", &circuits::random_circuit(12, 10, SEED));
}

#[test]
fn factored_random_12q_fused() {
    check_fused_vs_unfused(
        "random 12q d10 fused",
        &circuits::random_circuit(12, 10, SEED),
    );
}

// ===== hardware_efficient_ansatz =====

#[test]
fn factored_hea_12q_sv() {
    check_sv_cross(
        "hea 12q l3 sv",
        &circuits::hardware_efficient_ansatz(12, 3, SEED),
    );
}

#[test]
fn factored_hea_12q_fused() {
    check_fused_vs_unfused(
        "hea 12q l3 fused",
        &circuits::hardware_efficient_ansatz(12, 3, SEED),
    );
}

#[test]
fn factored_hea_16q_sv() {
    check_sv_cross(
        "hea 16q l2 sv",
        &circuits::hardware_efficient_ansatz(16, 2, SEED),
    );
}

// ===== clifford_heavy =====

#[test]
fn factored_clifford_heavy_12q_sv() {
    check_sv_cross(
        "clifford_heavy 12q d10 sv",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

#[test]
fn factored_clifford_heavy_12q_fused() {
    check_fused_vs_unfused(
        "clifford_heavy 12q d10 fused",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

// ===== clifford_random_pairs =====

#[test]
fn factored_clifford_random_pairs_12q_sv() {
    check_sv_cross(
        "clifford_random_pairs 12q d10 sv",
        &circuits::clifford_random_pairs(12, 10, SEED),
    );
}

#[test]
fn factored_clifford_random_pairs_12q_fused() {
    check_fused_vs_unfused(
        "clifford_random_pairs 12q d10 fused",
        &circuits::clifford_random_pairs(12, 10, SEED),
    );
}

// ===== independent_bell_pairs =====

#[test]
fn factored_bell_pairs_12q_sv() {
    check_sv_cross("bell_pairs 6 sv", &circuits::independent_bell_pairs(6));
}

#[test]
fn factored_bell_pairs_12q_fused() {
    check_fused_vs_unfused("bell_pairs 6 fused", &circuits::independent_bell_pairs(6));
}

#[test]
fn factored_bell_pairs_20q_sv() {
    check_sv_cross("bell_pairs 10 sv", &circuits::independent_bell_pairs(10));
}

#[test]
fn factored_bell_pairs_20q_fused() {
    check_fused_vs_unfused("bell_pairs 10 fused", &circuits::independent_bell_pairs(10));
}

// ===== independent_random_blocks =====

#[test]
fn factored_random_blocks_12q_sv() {
    check_sv_cross(
        "random_blocks 4x3 d5 sv",
        &circuits::independent_random_blocks(4, 3, 5, SEED),
    );
}

#[test]
fn factored_random_blocks_12q_fused() {
    check_fused_vs_unfused(
        "random_blocks 4x3 d5 fused",
        &circuits::independent_random_blocks(4, 3, 5, SEED),
    );
}

// The factored MultiFused path shares the statevector's tiered kernel, so it
// needs a block wide enough to reach every tier: local targets 0-13 land in L2,
// 14-16 in L3, and 17 above both. A 4-qubit block would exercise none of them.
#[test]
fn factored_multi_fused_spans_every_tier_20q_sv() {
    const WIDE: usize = 18;
    let mut c = Circuit::new(20, 0);
    for q in 0..WIDE {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..WIDE - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::H, &[18]);
    c.add_gate(Gate::Cx, &[18, 19]);
    for q in 0..WIDE {
        c.add_gate(Gate::Ry(0.19 + q as f64 * 0.07), &[q]);
    }

    assert_eq!(
        c.independent_subsystems().len(),
        2,
        "the wide block and the trailing pair must stay separate or nothing is factored"
    );
    assert!(
        has_multi_fused(&c),
        "the trailing rotation layer should batch into MultiFused"
    );
    // The SV cross alone cannot see a broken tiered kernel: both backends share
    // it. The unfused arm applies each rotation on its own, so it can.
    check_fused_vs_unfused("multi_fused all tiers 20q fused", &c);
    check_sv_cross("multi_fused all tiers 20q sv", &c);
}

#[test]
fn factored_random_blocks_20q_sv() {
    check_sv_cross(
        "random_blocks 5x4 d5 sv",
        &circuits::independent_random_blocks(5, 4, 5, SEED),
    );
}

#[test]
fn factored_random_blocks_20q_fused() {
    check_fused_vs_unfused(
        "random_blocks 5x4 d5 fused",
        &circuits::independent_random_blocks(5, 4, 5, SEED),
    );
}

// ===== ghz =====

#[test]
fn factored_ghz_12q_sv() {
    check_sv_cross("ghz 12q sv", &circuits::ghz_circuit(12));
}

#[test]
fn factored_ghz_12q_fused() {
    check_fused_vs_unfused("ghz 12q fused", &circuits::ghz_circuit(12));
}

// ===== qaoa =====

#[test]
fn factored_qaoa_12q_sv() {
    check_sv_cross("qaoa 12q l3 sv", &circuits::qaoa_circuit(12, 3, SEED));
}

#[test]
fn factored_qaoa_12q_fused() {
    check_fused_vs_unfused("qaoa 12q l3 fused", &circuits::qaoa_circuit(12, 3, SEED));
}

#[test]
fn factored_qaoa_16q_sv() {
    check_sv_cross("qaoa 16q l3 sv", &circuits::qaoa_circuit(16, 3, SEED));
}

// ===== single_qubit_rotation =====

#[test]
fn factored_single_qubit_rotation_12q_sv() {
    check_sv_cross(
        "single_qubit_rotation 12q d5 sv",
        &circuits::single_qubit_rotation_circuit(12, 5, SEED),
    );
}

#[test]
fn factored_single_qubit_rotation_12q_fused() {
    check_fused_vs_unfused(
        "single_qubit_rotation 12q d5 fused",
        &circuits::single_qubit_rotation_circuit(12, 5, SEED),
    );
}

// ===== clifford_t =====

#[test]
fn factored_clifford_t_12q_sv() {
    check_sv_cross(
        "clifford_t 12q d10 t=0.2 sv",
        &circuits::clifford_t_circuit(12, 10, 0.2, SEED),
    );
}

#[test]
fn factored_clifford_t_12q_fused() {
    check_fused_vs_unfused(
        "clifford_t 12q d10 t=0.2 fused",
        &circuits::clifford_t_circuit(12, 10, 0.2, SEED),
    );
}

// ===== w_state =====

#[test]
fn factored_w_state_8q_sv() {
    check_sv_cross("w_state 8q sv", &circuits::w_state_circuit(8));
}

#[test]
fn factored_w_state_8q_fused() {
    check_fused_vs_unfused("w_state 8q fused", &circuits::w_state_circuit(8));
}

// ===== quantum_volume =====

#[test]
fn factored_quantum_volume_8q_sv() {
    check_sv_cross(
        "quantum_volume 8q d2 sv",
        &circuits::quantum_volume_circuit(8, 2, SEED),
    );
}

#[test]
fn factored_quantum_volume_12q_fused() {
    check_fused_vs_unfused(
        "quantum_volume 12q d1 fused",
        &circuits::quantum_volume_circuit(12, 1, SEED),
    );
}

// ===== local_clifford_blocks =====

#[test]
fn factored_local_clifford_blocks_12q_sv() {
    check_sv_cross(
        "local_clifford_blocks 4x3 d10 sv",
        &circuits::local_clifford_blocks(4, 3, 10, SEED),
    );
}

#[test]
fn factored_local_clifford_blocks_12q_fused() {
    check_fused_vs_unfused(
        "local_clifford_blocks 4x3 d10 fused",
        &circuits::local_clifford_blocks(4, 3, 10, SEED),
    );
}

#[test]
fn factored_local_clifford_blocks_20q_sv() {
    check_sv_cross(
        "local_clifford_blocks 5x4 d10 sv",
        &circuits::local_clifford_blocks(5, 4, 10, SEED),
    );
}

#[test]
fn factored_local_clifford_blocks_20q_fused() {
    check_fused_vs_unfused(
        "local_clifford_blocks 5x4 d10 fused",
        &circuits::local_clifford_blocks(5, 4, 10, SEED),
    );
}

// ===== cz_chain =====

#[test]
fn factored_cz_chain_12q_sv() {
    check_sv_cross(
        "cz_chain 12q d5 sv",
        &circuits::cz_chain_circuit(12, 5, SEED),
    );
}

#[test]
fn factored_cz_chain_12q_fused() {
    check_fused_vs_unfused(
        "cz_chain 12q d5 fused",
        &circuits::cz_chain_circuit(12, 5, SEED),
    );
}

// ===== phase_estimation =====

#[test]
fn factored_phase_estimation_8q_sv() {
    check_sv_cross("qpe 8q sv", &circuits::phase_estimation_circuit(8));
}

#[test]
fn factored_phase_estimation_12q_fused() {
    check_fused_vs_unfused("qpe 12q fused", &circuits::phase_estimation_circuit(12));
}

// A factored register past 64 qubits has no dense probability vector and no
// representable lazy one either: the block mask is a u64 and Probabilities::len
// is 1 << total_qubits. The terminal must decline rather than shift past the
// word. Self-inverse bridges keep static analysis at one component so the sim
// runs the whole circuit on one backend instead of decomposing it.
#[test]
fn factored_oversize_probabilities_decline_instead_of_panicking() {
    use prism_q::sim::BackendKind;

    let n = 100;
    let block = 5;
    let mut circuit = Circuit::new(n, 0);
    for base in (0..n).step_by(block) {
        circuit.add_gate(Gate::H, &[base]);
        for q in base..(base + block - 1).min(n - 1) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    for base in (block..n).step_by(block) {
        circuit.add_gate(Gate::Cx, &[0, base]);
        circuit.add_gate(Gate::Cx, &[0, base]);
    }
    assert_eq!(
        circuit.independent_subsystems().len(),
        1,
        "bridges must leave one static component or the sim decomposes instead"
    );

    let out = prism_q::sim::simulate(&circuit)
        .backend(BackendKind::Factored)
        .seed(SEED)
        .run()
        .expect("an oversize register must not fail the run");
    assert!(
        out.probabilities.is_none(),
        "a 100 qubit factored register cannot serve a dense probability vector"
    );
}

// The multi-block probability terminal now returns per-block marginals instead
// of the merged 2^n vector. Expanding that lazy form must reproduce the merge
// exactly, which pins the block bit-order convention: each block's probs are
// indexed by its own qubits in ascending global order, and `mask` records which
// global positions those are.
#[test]
fn factored_block_probabilities_expand_to_the_merged_vector() {
    use prism_q::backend::Backend;
    use prism_q::sim::{BackendKind, Probabilities};

    for n in [10usize, 14, 16] {
        let circuit = circuits::partially_independent_circuit(n, 4, SEED);
        assert!(
            circuit.independent_subsystems().len() > 1,
            "{n}q: circuit must stay partially independent for this to test anything"
        );

        let mut backend = FactoredBackend::new(SEED);
        prism_q::sim::run_on(&mut backend, &circuit).unwrap();

        let lazy = backend
            .block_probabilities()
            .expect("multi-block state must offer per-block probabilities");
        match &lazy {
            Probabilities::Factored { blocks, .. } => {
                assert!(blocks.len() > 1, "{n}q: expected more than one block")
            }
            Probabilities::Dense(_) => panic!("{n}q: expected the factored variant"),
        }

        let merged = backend.probabilities().unwrap();
        let expanded = lazy.to_vec();
        assert_eq!(expanded.len(), merged.len(), "{n}q: length mismatch");
        for (i, (a, b)) in expanded.iter().zip(&merged).enumerate() {
            assert!(
                (a - b).abs() < FACTORED_EPS,
                "{n}q: state {i}: lazy {a} vs merged {b}"
            );
        }

        // The public run must now carry the lazy form end to end.
        let out = prism_q::sim::simulate(&circuit)
            .backend(BackendKind::Factored)
            .seed(SEED)
            .run()
            .unwrap();
        assert!(matches!(
            out.probabilities,
            Some(Probabilities::Factored { .. })
        ));
    }
}
