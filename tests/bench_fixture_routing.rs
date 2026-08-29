//! Bench fixtures reach the backend their row names.
//!
//! Independent-subsystem decomposition is chosen before a backend family is
//! resolved and does not consult an explicit [`BackendKind`], so a fixture whose
//! interaction graph splits runs on per-block backends no matter what a caller
//! asked for. That is correct routing and a silently wrong measurement: the
//! affected rows stayed green, kept reporting, and went non-monotonic in qubit
//! count. Nothing else in the suite checks the engine a run actually used.

use prism_q::backend::Backend;
use prism_q::sim::ResolvedBackend;
use prism_q::{BackendKind, MpsBackend, circuits, sim};

const SEED: u64 = 0xDEAD_BEEF;

#[track_caller]
fn assert_resolves(label: &str, kind: BackendKind, circuit: &prism_q::circuit::Circuit) {
    let outcome = sim::simulate(circuit)
        .backend(kind.clone())
        .seed(42)
        .run()
        .unwrap();
    assert!(
        !matches!(outcome.metadata.backend, ResolvedBackend::Decomposed),
        "{label}: asked for {kind:?}, ran on {:?}. The fixture's interaction \
         graph splits, so this row measures the decomposed route rather than \
         the backend it names.",
        outcome.metadata.backend
    );
}

// Widths 18 and 20 are the ones that regressed: the brick layer dropped a bond
// often enough that roughly half of all (n, seed) pairs split the register, and
// 22 and 24 happened to survive while 18 and 20 did not.
#[test]
fn random_circuit_rows_reach_their_backend() {
    for n in [12usize, 14, 16, 18, 20, 22, 24] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        assert_resolves(
            &format!("random_d10/{n}"),
            BackendKind::Statevector,
            &circuit,
        );
    }

    // The `factored/random_d10` group compares two arms on one circuit, so a
    // split sends both to the same route and the comparison measures nothing.
    for n in [16usize, 20, 24] {
        let circuit = circuits::random_circuit(n, 10, SEED);
        for kind in [
            BackendKind::Statevector,
            BackendKind::Factored,
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 64 },
        ] {
            assert_resolves(&format!("random_d10/{n}"), kind, &circuit);
        }
    }
}

// `scalability_d5` sweeps 2 to 26 at depth 5, so it is the row set most exposed
// to a fixture that connects only at higher depth.
#[test]
fn scalability_sweep_rows_reach_their_backend() {
    for n in (4..=26).step_by(2) {
        let circuit = circuits::random_circuit(n, 5, SEED);
        assert_resolves(
            &format!("scalability_d5/{n}"),
            BackendKind::Statevector,
            &circuit,
        );
    }
}

#[test]
fn diagonal_mixed_rows_reach_their_backend() {
    for n in [16usize, 20, 22, 24, 26] {
        let circuit = circuits::diagonal_mixed_circuit(n, 6, SEED);
        assert_resolves(
            &format!("diag_mixed_l6/{n}"),
            BackendKind::Statevector,
            &circuit,
        );
    }
}

// The `mps/brickwork_d24` rows price the bond cap, so what needs pinning is
// entanglement, not routing. The peak is measured through the instruction
// stream (a saturated cap proves the uncapped peak meets it) rather than
// trusted from construction: Cx on plus states holds dense_entanglement at
// bond 1.
#[test]
fn brickwork_rows_saturate_the_bond_ladder() {
    fn stream_peak_saturates(circuit: &prism_q::circuit::Circuit, cap: usize) -> bool {
        let mut backend = MpsBackend::new(SEED, cap);
        backend
            .init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        for instruction in &circuit.instructions {
            backend.apply(instruction).unwrap();
            if backend.current_max_bond_dim() >= cap {
                return true;
            }
        }
        false
    }

    let circuit = circuits::brickwork_circuit(18, 24, SEED);
    assert!(
        stream_peak_saturates(&circuit, 32),
        "brickwork_d24/18: the bottom ladder rung no longer truncates"
    );
    assert!(
        stream_peak_saturates(&circuit, 512),
        "brickwork_d24/18: the uncapped peak no longer clears the 256 cap, \
         so the chi ladder measures allocation and traversal instead of bond"
    );
}

// The `mps/matched_d12` row prices SWAP routing at real bond, so its two
// claims are that gates stay non-adjacent and that the bond saturates the
// cap. The second check is the one that catches a fixed pairing: its gates
// still count as non-adjacent, but routing parks each pair and drains the
// chain to bond 2.
#[test]
fn matched_rows_route_and_saturate() {
    let circuit = circuits::matched_brickwork_circuit(16, 12, SEED);
    let non_adjacent = circuit
        .instructions
        .iter()
        .filter(|instruction| {
            matches!(instruction, prism_q::Instruction::Gate { targets, .. }
                if targets.len() == 2 && targets[0].abs_diff(targets[1]) > 1)
        })
        .count();
    assert!(
        non_adjacent >= 50,
        "matched_d12/16: only {non_adjacent} non-adjacent gates; the row no \
         longer prices routing"
    );

    let mut backend = MpsBackend::new(SEED, 64);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for instruction in &circuit.instructions {
        backend.apply(instruction).unwrap();
        if backend.current_max_bond_dim() >= 64 {
            return;
        }
    }
    panic!("matched_d12/16: the bond never saturated the 64 cap");
}

#[test]
fn statevector_corpus_rows_reach_the_statevector() {
    for n in [16usize, 20] {
        assert_resolves(
            &format!("clifford_d10/{n}"),
            BackendKind::Statevector,
            &circuits::clifford_heavy_circuit(n, 10, SEED),
        );
        assert_resolves(
            &format!("qaoa_l3/{n}"),
            BackendKind::Statevector,
            &circuits::qaoa_circuit(n, 3, SEED),
        );
        assert_resolves(
            &format!("hea_l5/{n}"),
            BackendKind::Statevector,
            &circuits::hardware_efficient_ansatz(n, 5, SEED),
        );
        assert_resolves(
            &format!("qv/{n}"),
            BackendKind::Statevector,
            &circuits::quantum_volume_circuit(n, n, SEED),
        );
        assert_resolves(
            &format!("qpe_t_gate/{n}"),
            BackendKind::Statevector,
            &circuits::phase_estimation_circuit(n),
        );
    }
}
