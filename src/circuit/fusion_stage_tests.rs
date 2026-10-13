use super::*;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::fusion::{fuse_staged, fuse_unstaged};
use crate::circuits;
use crate::sim::unified_pauli::PauliTerm;

const SEED: u64 = 42;

fn ring_matching(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    let mut edges: Vec<(usize, usize)> = (0..n).map(|q| (q, (q + 1) % n)).collect();
    edges.extend((0..n / 2).map(|k| (k, n - 1 - k)));
    for layer in 0..3 {
        for (e, &(a, b)) in edges.iter().enumerate() {
            c.add_gate(Gate::Rzz(0.3 + 0.01 * (layer * 31 + e) as f64), &[a, b]);
        }
        for q in 0..n {
            c.add_gate(Gate::Rx(0.7 + 0.02 * (layer * 17 + q) as f64), &[q]);
        }
    }
    c
}

fn trotter(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for (coefficient, factors) in &circuits::jordan_wigner_hamiltonian(n, 120, SEED) {
        if !factors.is_empty() {
            c.add_pauli_rotation(0.1 * coefficient, factors);
        }
    }
    c
}

fn mixed(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 1);
    c.instructions = circuits::hardware_efficient_ansatz(n, 2, SEED).instructions;
    c.add_pauli_rotation(
        0.4,
        &[PauliTerm::x(0), PauliTerm::z(n / 2), PauliTerm::y(n - 1)],
    );
    c.add_gate(Gate::Swap, &[1, n - 2]);
    c.add_gate(Gate::cphase(0.3), &[2, n - 1]);
    c.add_gate(Gate::Cz, &[0, n - 3]);
    c.add_measure(n / 2, 0);
    c.add_gate(Gate::Rx(0.9), &[n / 2]);
    c.add_reset(1);
    c.add_gate(Gate::Ry(0.2), &[1]);
    c.instructions.push(Instruction::Barrier {
        qubits: smallvec![0, 1],
    });
    for q in 0..n - 1 {
        c.add_gate(Gate::Rzz(0.1 * q as f64), &[q, q + 1]);
        c.add_gate(Gate::Ry(0.05 * q as f64), &[q]);
    }
    c
}

fn corpus(n: usize) -> Vec<(&'static str, Circuit)> {
    vec![
        ("hea_l5", circuits::hardware_efficient_ansatz(n, 5, SEED)),
        ("random_d10", circuits::random_circuit(n, 10, SEED)),
        ("qaoa_l3", circuits::qaoa_circuit(n, 3, SEED)),
        ("ring_matching", ring_matching(n)),
        ("qv", circuits::quantum_volume_circuit(n, 4, SEED)),
        (
            "diag_mixed_l6",
            circuits::diagonal_mixed_circuit(n, 6, SEED),
        ),
        ("qft", circuits::qft_circuit(n)),
        ("qpe", circuits::phase_estimation_circuit(n)),
        ("trotter", trotter(n)),
        ("mixed", mixed(n)),
    ]
}

fn run(circuit: &Circuit, instructions: &[Instruction]) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(SEED);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(instructions).unwrap();
    backend.state_vector().to_vec()
}

fn assert_staged_matches_unfused(circuit: &Circuit, geometry: TileGeometry, what: &str) {
    let staged = fuse_staged(circuit, geometry);
    let expected = run(circuit, &circuit.instructions);
    let got = run(circuit, &staged.instructions);
    for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
        assert!((a - b).norm() < 1e-10, "{what}: amplitude {i}, {a} vs {b}");
    }
}

#[test]
fn staged_streams_match_the_unfused_state_on_narrow_tiles() {
    for (n, low, budget) in [(10, 2, 3), (11, 3, 2), (12, 1, 4), (12, 4, 6)] {
        let geometry = TileGeometry::new(n, low, budget);
        for (name, circuit) in corpus(n) {
            assert_staged_matches_unfused(
                &circuit,
                geometry,
                &format!("{name}/{n} low {low} budget {budget}"),
            );
        }
    }
}

#[test]
fn staged_streams_match_the_unfused_state_on_the_host_tile() {
    let n = 20;
    for (name, circuit) in corpus(n) {
        assert_staged_matches_unfused(&circuit, TileGeometry::for_width(n), &format!("{name}/{n}"));
    }
}

#[test]
fn every_stage_fits_its_tile() {
    let (n, low, budget) = (12, 2, 3);
    let geometry = TileGeometry::new(n, low, budget);
    for (name, circuit) in corpus(n) {
        for inst in &fuse_staged(&circuit, geometry).instructions {
            let Instruction::Gate { gate, targets } = inst else {
                continue;
            };
            let needed = match gate {
                Gate::Multi2q(_) | Gate::MultiFused(_) => qubit_mask(targets),
                Gate::MultiPauliRot(data) => data
                    .rotations
                    .iter()
                    .fold(0u64, |m, &(x, _, _)| m | x as u64),
                _ => continue,
            };
            assert!(
                (needed & geometry.high).count_ones() <= budget,
                "{name}: {gate:?} on {targets:?} overflows the tile"
            );
        }
    }
}

#[test]
fn staging_is_a_function_of_the_stream_alone() {
    let n = 12;
    let geometry = TileGeometry::new(n, 2, 3);
    for (name, circuit) in corpus(n) {
        let first = fuse_staged(&circuit, geometry).into_owned();
        let second = fuse_staged(&circuit.clone(), geometry).into_owned();
        assert_eq!(
            format!("{:?}", first.instructions),
            format!("{:?}", second.instructions),
            "{name}"
        );
    }
}

#[test]
fn staging_leaves_a_stream_of_lone_gates_borrowed() {
    let mut c = Circuit::new(20, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 19]);
    c.add_gate(Gate::H, &[19]);
    let staged = plan_with(
        Cow::Borrowed(&c),
        TileGeometry::new(20, 6, 0),
        &mut Tracer::off(),
    );
    assert!(matches!(staged, Cow::Borrowed(_)));
}

#[test]
fn the_planner_takes_fewer_passes_than_the_batching_passes_on_its_target_shapes() {
    let n = 24;
    let geometry = TileGeometry::for_width(n);
    for (name, circuit) in [
        (
            "hea_l5",
            circuits::hardware_efficient_ansatz(n, 5, 0xDEAD_BEEF),
        ),
        ("qaoa_l3", circuits::qaoa_circuit(n, 3, 0xDEAD_BEEF)),
        ("qv", circuits::quantum_volume_circuit(n, n, 0xDEAD_BEEF)),
    ] {
        let staged = fuse_staged(&circuit, geometry);
        let unstaged = fuse_unstaged(&circuit, n);
        let passes = |c: &Circuit| {
            crate::backend::statevector::sweeps::sweep_profile(&c.instructions, n, 1).passes()
        };
        assert!(
            passes(&staged) < passes(&unstaged),
            "{name}: {} staged passes against {}",
            passes(&staged),
            passes(&unstaged)
        );
    }
}
