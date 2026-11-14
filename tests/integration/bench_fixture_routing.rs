//! Bench fixtures reach the backend their row names.
//!
//! Independent-subsystem decomposition is chosen before a backend family is
//! resolved, and only an explicit `FactoredStabilizer` keeps a splitting circuit
//! whole, so a fixture whose interaction graph splits runs on per-block backends
//! under any other explicit [`BackendKind`]. That is correct routing and a
//! silently wrong measurement: the affected rows stayed green, kept reporting,
//! and went non-monotonic in qubit count. Nothing else in the suite checks the engine a run actually used.

use crate::bench_fixtures;
use prism_q::backend::Backend;
use prism_q::sim::{Engine, ResolvedBackend, RouteRule};
use prism_q::{BackendKind, MpsBackend, NoiseChannel, NoiseEvent, NoiseModel, circuits, sim};

// The bench seed, not the test seed: these fixtures are the bench rows, and
// the routing under check is the routing the bench measures.
const SEED: u64 = 0xDEAD_BEEF;

#[test]
fn terminal_disjoint_rows_keep_ten_qubit_statevector_blocks() {
    for n in [60, 70, 100, 200] {
        for strided in [false, true] {
            let circuit = bench_fixtures::terminal_disjoint_blocks(n, strided, SEED);
            let components = circuit.independent_subsystems();
            assert_eq!(components.len(), n / 10);
            assert!(components.iter().all(|block| block.len() == 10));
            assert!(circuit.has_terminal_measurements_only());
            let result = sim::simulate(&circuit).seed(42).shots(4).unwrap();
            assert_eq!(result.metadata.backend, ResolvedBackend::Decomposed);
            assert_eq!(
                result.metadata.blocks,
                vec![ResolvedBackend::Statevector; n / 10]
            );
            let rule = if n > 64 {
                RouteRule::PerShot
            } else {
                RouteRule::IndependentBlocks
            };
            assert_eq!(result.metadata.reason, rule);
            for query in [sim::Query::Shots(4), sim::Query::Counts(4)] {
                let plan = sim::simulate(&circuit).seed(42).plan(query).unwrap();
                assert_eq!(plan.backend, result.metadata.backend);
                assert_eq!(plan.rule, rule);
            }
        }
    }
}

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

// `density_matrix/rzz_layers_fused` is the one density-matrix row that runs
// through `simulate`, so it is the one that can be claimed by the decomposed
// route instead. 10 qubits alone: the fixture is a line graph at every width,
// and a 12-qubit mixture costs seconds per run.
#[test]
fn density_matrix_fused_row_reaches_its_backend() {
    let circuit = circuits::qaoa_circuit(10, 6, SEED);
    assert_resolves("rzz_layers_fused/10", BackendKind::DensityMatrix, &circuit);
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

// The `mps/hotspots/measure_brickwork_16q_r3` row prices a measurement at
// bond, which its neighbour `measure_reset_32q_r3` cannot: a controlled-not on
// two plus states is the identity, so that fixture never leaves bond one. What
// this one owes is real bond around its measurement rounds.
#[test]
fn the_measured_brickwork_row_carries_bond() {
    let mut circuit = prism_q::circuit::Circuit::new(16, 16);
    for round in 0..3 {
        let layers = circuits::brickwork_circuit(16, 5, SEED);
        circuit.instructions.extend(layers.instructions);
        for q in (round % 2..16).step_by(4) {
            circuit.add_measure(q, q);
        }
    }

    let mut backend = MpsBackend::new(SEED, 64);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    let mut peak = 1usize;
    let mut measured = 0usize;
    for instruction in &circuit.instructions {
        backend.apply(instruction).unwrap();
        peak = peak.max(backend.current_max_bond_dim());
        if matches!(instruction, prism_q::Instruction::Measure { .. }) {
            measured += 1;
        }
    }
    assert!(
        peak >= 16,
        "measure_brickwork_16q_r3: peak bond {peak} leaves the row pricing bookkeeping"
    );
    assert_eq!(measured, 12, "measure_brickwork_16q_r3: measurement count");
}

// The `mps/brickwork_d24/b256/12` row prices the two-site update at bond with
// neither the cap nor the orthogonality-center walk, so its fixture has to stay
// under the cap. The half of that claim which reads the rank at which a
// threshold cut takes the center is held against the constant itself, in the
// backend's own tests; what an outside test can see is the cap.
#[test]
fn the_ungauged_row_stays_under_its_cap() {
    let circuit = circuits::brickwork_circuit(12, 24, SEED);
    let mut backend = MpsBackend::new(SEED, 256);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();

    let peak = backend.current_max_bond_dim();
    assert!(
        peak < 256,
        "brickwork_d24/12: peak bond {peak} reaches its own cap"
    );
    let discarded = backend.truncation_discarded();
    assert!(
        discarded < 1e-20,
        "brickwork_d24/12: the cap discarded {discarded:.3e}, so the row prices truncation"
    );
}

// The `mps/observables` rows price the environment sweeps over the finished
// chain, so what needs pinning is the bond that chain ends at, not the peak
// along the way.
#[test]
fn the_observable_rows_sweep_at_the_cap() {
    let circuit = circuits::brickwork_circuit(64, 12, SEED);
    let mut backend = MpsBackend::new(SEED, 64);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    assert_eq!(
        backend.current_max_bond_dim(),
        64,
        "observables/*_b64/64: the finished chain no longer sits at the cap"
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

// The sparse walk rows price map-walk kernels at a pinned entry count, and
// the densify rows trace the load-factor crossover against the dense arm, so
// the claims are that the register does not split (the decomposed route
// claimed the family these rows replaced) and that the map holds exactly 2^k
// entries once the H prefix has run: every later gate scales amplitudes in
// place or permutes keys one to one. The count is checked through the fused
// stream the bench executes, so a fusion change that starts branching or
// draining the map fails here.
fn sparse_entry_peak_and_final(circuit: &prism_q::circuit::Circuit) -> (usize, usize) {
    let fused = prism_q::circuit::fusion::fuse_circuit(circuit, true);
    let mut backend = prism_q::SparseBackend::new(SEED);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    let mut peak = 0;
    for instruction in &fused.instructions {
        backend.apply(instruction).unwrap();
        peak = peak.max(backend.entry_count());
    }
    (peak, backend.entry_count())
}

#[test]
fn sparse_walk_rows_pin_route_and_entry_count() {
    for (n, depth) in [(32usize, 5usize), (48, 5), (64, 5), (32, 2), (64, 2)] {
        let circuit = circuits::sparse_walk_circuit(n, 12, depth, SEED);
        assert_resolves(
            &format!("walk_k12 shape {n}/d{depth}"),
            BackendKind::Sparse,
            &circuit,
        );
        let (peak, last) = sparse_entry_peak_and_final(&circuit);
        assert_eq!(
            (peak, last),
            (1 << 12, 1 << 12),
            "walk_k12 {n}/d{depth}: the map no longer holds a pinned 4096 entries"
        );
    }
}

#[test]
fn sparse_densify_rows_pin_entry_ladder() {
    for k in [8usize, 14, 20] {
        let circuit = circuits::sparse_walk_circuit(20, k, 2, SEED);
        let (peak, last) = sparse_entry_peak_and_final(&circuit);
        assert_eq!(
            (peak, last),
            (1 << k, 1 << k),
            "densify k={k}: the ladder no longer walks the map to 2^k entries"
        );
    }
    // Route is set by the layer stream, which is identical across k.
    let circuit = circuits::sparse_walk_circuit(20, 8, 2, SEED);
    assert_resolves("densify/map/8", BackendKind::Sparse, &circuit);
    assert_resolves("densify/dense/8", BackendKind::Statevector, &circuit);
}

// `sparse/noisy_walk_k12` prices trajectory noise at a bounded map. Trajectories
// apply the stream raw so noise events stay aligned to it, so the pin reads the
// unfused stream. The damping row's bound is checked at its worst case: a jump
// after every gate on every target the jump can reach, which only ever merges
// or drops entries.
#[test]
fn sparse_noisy_walk_rows_stay_bounded_on_the_map() {
    use num_complex::Complex64;

    let zero = Complex64::new(0.0, 0.0);
    for n in [32usize, 64] {
        let circuit = circuits::sparse_walk_circuit(n, 12, 2, SEED);

        let mut backend = prism_q::SparseBackend::new(SEED);
        backend.init(n, 0).unwrap();
        let mut peak = 0;
        for instruction in &circuit.instructions {
            backend.apply(instruction).unwrap();
            peak = peak.max(backend.entry_count());
        }
        assert_eq!(
            (peak, backend.entry_count()),
            (1 << 12, 1 << 12),
            "noisy_walk_k12/{n}: the raw stream no longer holds a pinned 4096 entries"
        );

        let mut backend = prism_q::SparseBackend::new(SEED);
        backend.init(n, 0).unwrap();
        let mut peak = 0;
        for instruction in &circuit.instructions {
            backend.apply(instruction).unwrap();
            let prism_q::Instruction::Gate { targets, .. } = instruction else {
                continue;
            };
            for &q in targets.iter() {
                let p1 = backend.qubit_probability(q).unwrap();
                if p1 > 1e-12 {
                    let jump = [[zero, Complex64::new(p1.sqrt().recip(), 0.0)], [zero, zero]];
                    backend.apply_1q_matrix(q, &jump).unwrap();
                }
                peak = peak.max(backend.entry_count());
            }
        }
        assert!(
            peak <= 1 << 12,
            "noisy_walk_k12/{n}: the damping jump grew the map to {peak} entries"
        );

        let mut measured = circuit.clone();
        measured.measure_all();
        for (label, noise) in [
            (
                "amplitude_damping",
                NoiseModel::with_amplitude_damping(&measured, 0.01),
            ),
            (
                "depolarizing",
                NoiseModel::uniform_depolarizing(&measured, 0.01),
            ),
        ] {
            let shots = sim::simulate(&measured)
                .backend(BackendKind::Sparse)
                .noise(&noise)
                .seed(SEED)
                .shots(4)
                .unwrap();
            assert_eq!(
                shots.metadata.backend,
                ResolvedBackend::Sparse,
                "noisy_walk_k12/{label}/{n}"
            );
        }
    }
}

// `sparse/sampling` prices shot conversion on a near-empty map; the GHZ
// chain keeps the register connected, unlike the split fixture it replaced.
#[test]
fn sparse_sampling_fixture_reaches_the_sparse_backend() {
    for n in [24usize, 64] {
        let circuit = circuits::ghz_circuit(n);
        assert_resolves(&format!("sampling/{n}"), BackendKind::Sparse, &circuit);
        let (peak, last) = sparse_entry_peak_and_final(&circuit);
        assert_eq!(
            (peak, last),
            (2, 2),
            "sampling/{n}: the empty-map control stopped being empty"
        );
    }
}

#[test]
fn native_counts_rows_count_on_their_backend() {
    fn measure_all(circuit: &prism_q::circuit::Circuit) -> prism_q::circuit::Circuit {
        let mut measured = prism_q::circuit::Circuit::new(circuit.num_qubits, circuit.num_qubits);
        measured.instructions = circuit.instructions.clone();
        for q in 0..circuit.num_qubits {
            measured.add_measure(q, q);
        }
        measured
    }

    let mut dense_d4 = prism_q::circuit::Circuit::new(24, 0);
    for _ in 0..4 {
        for q in 0..24 {
            dense_d4.add_gate(prism_q::gates::Gate::H, &[q]);
        }
        for q in 0..23 {
            dense_d4.add_gate(prism_q::gates::Gate::Cx, &[q, q + 1]);
        }
    }

    let rows = [
        (
            "mps/dense_d4_24",
            BackendKind::Mps { max_bond_dim: 32 },
            ResolvedBackend::Mps,
            measure_all(&dense_d4),
        ),
        (
            "sparse/ghz_64",
            BackendKind::Sparse,
            ResolvedBackend::Sparse,
            measure_all(&circuits::ghz_circuit(64)),
        ),
        (
            "sparse/walk_k12_64",
            BackendKind::Sparse,
            ResolvedBackend::Sparse,
            measure_all(&circuits::sparse_walk_circuit(64, 12, 2, SEED)),
        ),
    ];
    for (label, kind, expected, circuit) in rows {
        let counts = sim::simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .sample_counts(64)
            .unwrap();
        assert_eq!(counts.metadata.backend, expected, "native_counts/{label}");
    }
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
    assert_resolves(
        "entanglement_16q_d10/sparse",
        BackendKind::Statevector,
        &bench_fixtures::sparse_entanglement_circuit(16, 10),
    );
}

// The `statevector/dense_unitary` rows price the dense k-qubit kernel. Two layers of
// block tilings connect the register only through the shift between them, and the
// fused stream must still hold every block as a `Gate::Unitary` of width k: a lowering
// to `Mcu` or a diagonal batch, or a fusion pass that starts absorbing the blocks,
// would leave the rows timing a different kernel.
#[test]
fn dense_unitary_rows_keep_every_block_dense() {
    use prism_q::circuit::Instruction;
    use prism_q::gates::Gate;

    for k in [3usize, 4] {
        for n in [20usize, 24] {
            let layers = bench_fixtures::DENSE_UNITARY_LAYERS;
            let circuit = bench_fixtures::dense_unitary_layers(n, k, layers, SEED);
            let label = format!("dense_unitary/k{k}/{n}");
            assert_resolves(&label, BackendKind::Statevector, &circuit);

            let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
            let dense = fused
                .instructions
                .iter()
                .filter(|inst| {
                    matches!(
                        inst,
                        Instruction::Gate {
                            gate: Gate::Unitary(_),
                            targets,
                        } if targets.len() == k
                    )
                })
                .count();
            assert_eq!(
                (dense, fused.instructions.len()),
                (n / k * layers, n / k * layers),
                "{label}: the fused stream is no longer the dense blocks alone"
            );
        }
    }
}

// `product/expectation` and `product/noisy_trajectory` price the product backend's own
// terminals, so each must report it. A circuit with no entangling gate is the one the
// decomposed route would split per qubit, and a marginal route that fell back to the
// dense probability vector would fail outright at these widths rather than run slow.
#[test]
fn product_expectation_rows_run_on_the_product_backend() {
    for n in [64usize, 128, 256, 512] {
        let circuit = circuits::single_qubit_rotation_circuit(n, 10, SEED);
        let observables = bench_fixtures::product_observables(n);

        let expectations = sim::simulate(&circuit)
            .backend(BackendKind::ProductState)
            .seed(42)
            .expectation_values_reported(&observables)
            .unwrap();
        assert_eq!(
            expectations.metadata.backend,
            ResolvedBackend::ProductState,
            "expectation/pauli/{n}"
        );
        assert_eq!(expectations.values.len(), 2 * n - 1);

        let marginals = sim::simulate(&circuit)
            .backend(BackendKind::ProductState)
            .seed(42)
            .marginals()
            .unwrap();
        assert_eq!(
            marginals.metadata.backend,
            ResolvedBackend::ProductState,
            "expectation/marginals/{n}"
        );

        let mut measured = circuit.clone();
        measured.measure_all();
        let noise = NoiseModel::uniform_depolarizing(&measured, 0.01);
        let shots = sim::simulate(&measured)
            .backend(BackendKind::ProductState)
            .noise(&noise)
            .seed(SEED)
            .shots(4)
            .unwrap();
        assert_eq!(
            shots.metadata.backend,
            ResolvedBackend::ProductState,
            "noisy_trajectory/{n}"
        );
    }
}

// The `factored/disjoint_blocks` rows price the collapse a batched diagonal
// gate causes when it merges its whole target list. Two properties carry that,
// and neither is visible in a timing: the fused stream must hold a batch whose
// targets span the register, and the run must end with one block per ring.
// Raising the diagonal-batch floor past these widths would leave plain `Rzz`
// rows measuring nothing, still green.
#[test]
fn disjoint_block_rows_pin_the_batch_and_the_block_count() {
    use prism_q::backend::Backend;
    use prism_q::backend::factored::FactoredBackend;
    use prism_q::circuit::Instruction;
    use prism_q::gates::Gate;
    use prism_q::sim::Probabilities;

    for n in [20usize, 24] {
        let circuit = circuits::disjoint_block_layers_circuit(n, n / 2, 4, SEED);
        let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);

        let spanning_batches = fused
            .instructions
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    Instruction::Gate {
                        gate: Gate::BatchRzz(_) | Gate::DiagonalBatch(_),
                        targets,
                    } if targets.len() == n
                )
            })
            .count();
        assert!(
            spanning_batches > 0,
            "disjoint_blocks/{n}: no batched diagonal gate spans the register, \
             so the row prices nothing"
        );

        let mut backend = FactoredBackend::new(SEED);
        prism_q::sim::run_on(&mut backend, &circuit).unwrap();
        match backend.block_probabilities() {
            Some(Probabilities::Factored { blocks, .. }) => assert_eq!(
                blocks.len(),
                2,
                "disjoint_blocks/{n}: the two rings must stay separate blocks"
            ),
            _ => panic!("disjoint_blocks/{n}: the run collapsed to a single block"),
        }
    }
}

// `factored/measure_split` prices what a mid-circuit collapse leaves behind on
// the factored backend, so its rows only mean something if a plain run lands
// there. The fixture's pair keeps the largest component at `n - 2`, the width
// the decomposition heuristic declines, and the no-collapse control must route
// the same way or it controls for a different engine. The widest block at the
// end separates the two: the collapse leaves the survivors alone, the control
// keeps the whole chain.
#[test]
fn measure_split_rows_reach_the_factored_backend() {
    for n in [16usize, 20] {
        let circuit = circuits::measure_split_circuit(n, 24, SEED);
        let mut control = circuit.clone();
        control.instructions.retain(|inst| {
            !matches!(
                inst,
                prism_q::Instruction::Measure { .. } | prism_q::Instruction::Reset { .. }
            )
        });

        for (label, circ) in [("measure_split", &circuit), ("no_collapse", &control)] {
            let outcome = sim::simulate(circ).seed(42).run().unwrap();
            assert_eq!(
                outcome.metadata.backend,
                ResolvedBackend::Factored,
                "{label}/{n}: Auto must pick the factored backend"
            );
            for kind in [BackendKind::Factored, BackendKind::Statevector] {
                assert_resolves(&format!("{label}/{n}"), kind, circ);
            }
        }

        for (label, circ, widest) in [
            ("measure_split", &circuit, (n - 2) / 2),
            ("no_collapse", &control, n - 2),
        ] {
            let mut backend = prism_q::FactoredBackend::new(SEED);
            sim::run_on(&mut backend, circ).unwrap();
            let Some(prism_q::sim::Probabilities::Factored { blocks, .. }) =
                backend.block_probabilities()
            else {
                panic!("{label}/{n}: the run collapsed to a single block");
            };
            let max = blocks.iter().map(|b| b.mask.count_ones() as usize).max();
            assert_eq!(
                max,
                Some(widest),
                "{label}/{n}: the layers must run on a block of {widest} qubits"
            );
        }
    }
}

// `factored/sampling` prices the factored backend's own draw over its
// sub-states. The decomposed route would sample per block on separate
// backends and never call it, and a run that merged the pair into the chain
// would leave one block and price the single-block case instead.
#[test]
fn factored_sampling_rows_draw_from_two_sub_states() {
    for n in [16usize, 20, 24] {
        let circuit = circuits::partially_independent_circuit(n, 5, SEED);
        let mut measured = circuit.clone();
        measured.measure_all();

        let shots = sim::simulate(&measured)
            .backend(BackendKind::Factored)
            .seed(SEED)
            .shots(1_000)
            .unwrap();
        assert_eq!(
            shots.metadata.backend,
            ResolvedBackend::Factored,
            "factored/sampling/{n}q: the shots ran on {:?}",
            shots.metadata.backend
        );

        let mut backend = prism_q::FactoredBackend::new(SEED);
        sim::run_on(&mut backend, &circuit).unwrap();
        let Some(prism_q::sim::Probabilities::Factored { blocks, .. }) =
            backend.block_probabilities()
        else {
            panic!("factored/sampling/{n}q: the run collapsed to a single block");
        };
        let mut widths: Vec<u32> = blocks.iter().map(|b| b.mask.count_ones()).collect();
        widths.sort_unstable();
        assert_eq!(
            widths,
            vec![2, n as u32 - 2],
            "factored/sampling/{n}q: the sampler must draw over the pair and the chain"
        );
    }
}

// The `fstab_*` rows of `auto/crossover` bracket the factored-stabilizer floor,
// so they owe a route on each side of it: per-block tableaux under 16 qubits,
// one factored-stabilizer backend from 16 up. Above the floor the `auto` arm
// must also return what the `factored` arm's `run_on` returns, or the pair
// prices two different answers rather than the routing work between them.
#[test]
fn fstab_crossover_rows_straddle_the_floor() {
    let grid = [
        (2usize, 16usize),
        (3, 16),
        (4, 16),
        (6, 16),
        (8, 16),
        (10, 16),
        (2, 8),
        (3, 8),
        (4, 8),
        (8, 8),
        (12, 8),
        (20, 8),
        (3, 6),
        (4, 6),
        (8, 6),
        (4, 4),
        (5, 4),
        (6, 4),
        (8, 4),
        (16, 4),
        (24, 4),
        (40, 4),
        (6, 3),
        (8, 3),
        (16, 3),
        (8, 2),
        (10, 2),
        (12, 2),
        (16, 2),
        (32, 2),
        (48, 2),
        (80, 2),
        (4, 2),
        (6, 2),
        (3, 4),
        (2, 6),
        (2, 4),
        (7, 2),
    ];
    let mut below = 0;
    for (blocks, block_size) in grid {
        let n = blocks * block_size;
        let label = format!("fstab_{n}q_b{block_size}");
        let circuit = circuits::local_clifford_blocks(blocks, block_size, 200, SEED);
        let auto = sim::simulate(&circuit).seed(42).run().unwrap();

        if n < 16 {
            below += 1;
            assert_eq!(
                auto.metadata.backend,
                ResolvedBackend::Decomposed,
                "{label}: under the floor Auto must run per block"
            );
            continue;
        }
        assert_eq!(
            auto.metadata.backend,
            ResolvedBackend::FactoredStabilizer,
            "{label}: from the floor up Auto must run one factored-stabilizer backend"
        );
        let mut backend = prism_q::FactoredStabilizerBackend::new(42);
        let mut direct = sim::run_on(&mut backend, &circuit).unwrap();
        assert_eq!(
            auto.metadata.reason,
            prism_q::RouteRule::LargeCliffordBlocks
        );
        direct.metadata.reason = auto.metadata.reason;
        assert_eq!(
            format!("{:?}", auto.metadata),
            format!("{:?}", direct.metadata),
            "{label}"
        );
        assert_eq!(auto.classical_bits, direct.classical_bits, "{label}");
        if n <= 64 {
            assert_eq!(
                auto.probabilities.map(|p| p.marginals()),
                direct.probabilities.map(|p| p.marginals()),
                "{label}"
            );
        }
    }
    assert!(below >= 6, "the grid lost its rows under the floor");
}

// `stabilizer/random_pairs` prices the batched cross-word kernel, whose buffer
// takes only pairs split across two tableau words and runs the batched form
// once four of them are held. That the fixture feeds it is what an external
// test can see. Which widths reach the route the kernel lives on is set by the
// sparse gate index and pinned in the crate's own stabilizer tests, where that
// predicate is in scope.
fn peak_buffered_cross_word(circuit: &prism_q::circuit::Circuit) -> usize {
    let mut held = vec![false; circuit.num_qubits];
    let mut buffered = 0usize;
    let mut peak = 0usize;

    for instruction in &circuit.instructions {
        let prism_q::Instruction::Gate { targets, .. } = instruction else {
            peak = peak.max(buffered);
            buffered = 0;
            held.fill(false);
            continue;
        };
        if targets.iter().any(|&q| held[q]) {
            peak = peak.max(buffered);
            buffered = 0;
            held.fill(false);
        }
        if targets.len() == 2 && targets[0] / 64 != targets[1] / 64 {
            buffered += 1;
            for &q in targets {
                held[q] = true;
            }
        }
    }
    peak.max(buffered)
}

#[test]
fn random_pairs_rows_buffer_cross_word_gates() {
    for n in [500usize, 1000] {
        let circuit = circuits::clifford_random_pairs(n, 10, SEED);
        assert_resolves(
            &format!("random_pairs/{n}"),
            BackendKind::Stabilizer,
            &circuit,
        );

        let peak = peak_buffered_cross_word(&circuit);
        assert!(
            peak >= 4,
            "random_pairs/{n}: a flush holds at most {peak} cross-word gates, \
             under the four the batched kernel takes over at"
        );
    }
}

// A model the Clifford samplers refuse is not an error: it runs on the
// trajectory engine or on per-shot replay, one state per shot, and returns
// shots either way, so a demoted row keeps reporting while measuring an engine
// it does not name. The depth ratio keeps the three `noisy_sampling/compiled_*`
// models and the `noisy_sampling/low_rank/*` rows off the frame sampler.
#[test]
fn noisy_sampling_rows_take_the_compiled_sampler() {
    let mut circuit = circuits::clifford_heavy_circuit(100, 10, SEED);
    circuit.num_classical_bits = circuit.num_qubits;
    for q in 0..circuit.num_qubits {
        circuit.add_measure(q, q);
    }

    let pauli = NoiseModel::uniform_depolarizing(&circuit, 0.001);

    let mut pair = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    for (events, instruction) in pair.after_gate.iter_mut().zip(&circuit.instructions) {
        if let prism_q::Instruction::Gate { targets, .. } = instruction
            && targets.len() == 2
        {
            events.push(NoiseEvent {
                channel: NoiseChannel::TwoQubitDepolarizing { p: 0.01 },
                qubits: targets.iter().copied().collect(),
            });
        }
    }

    let mut readout = NoiseModel::uniform_depolarizing(&circuit, 0.001);
    readout.with_readout_error(0.02, 0.05);

    // 10_000 shots, as the rows run.
    for (label, model) in [
        ("compiled_pauli", &pauli),
        ("compiled_pair", &pair),
        ("compiled_readout", &readout),
    ] {
        let result = sim::simulate(&circuit)
            .noise(model)
            .seed(SEED)
            .shots(10_000)
            .unwrap();
        assert_eq!(
            result.metadata.engine,
            Some(Engine::NoisyCompiledSampler),
            "{label}: the model ran on {:?}, not on the compiled sampler",
            result.metadata.backend
        );
    }

    for n in [12usize, 16] {
        let mut circuit = circuits::clifford_heavy_circuit(n, 10, SEED);
        circuit.num_classical_bits = n;
        for q in 0..n {
            circuit.add_measure(q, q);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let result = sim::simulate(&circuit)
            .noise(&noise)
            .seed(SEED)
            .shots(10_000)
            .unwrap();
        assert_eq!(
            result.metadata.engine,
            Some(Engine::NoisyCompiledSampler),
            "low_rank/clifford_{n}q_10k"
        );
    }
}

// `dynamic/temporal_prefix_blocks` prices the per-shot loop for a circuit that
// splits into blocks and also opens on a Clifford prefix, which runs each shot
// through the decomposed route block by block rather than the decomposed shot
// loop. Three properties put it there and none shows in a timing: the shots
// run decomposed, the leading Clifford run reaches the temporal-prefix floor of
// `max(2n, 16)` gates for the whole register, and it reaches 16 on every block,
// so each block runs its own Clifford prefix too.
#[test]
fn temporal_prefix_block_rows_split_and_open_on_a_clifford_prefix() {
    use prism_q::Instruction;

    for block_size in [6usize, 8] {
        let circuit = circuits::clifford_prefix_measured_blocks(2, block_size, 4, SEED);
        let n = circuit.num_qubits;
        assert!(!circuit.is_clifford_only());

        let leading: Vec<&[usize]> = circuit
            .instructions
            .iter()
            .map_while(|inst| match inst {
                Instruction::Gate { gate, targets } if gate.is_clifford() => Some(&targets[..]),
                _ => None,
            })
            .collect();
        assert!(
            leading.len() >= (2 * n).max(16),
            "{n}q: {} leading Clifford gates, under the temporal-prefix floor",
            leading.len()
        );
        assert!(leading.len() < circuit.instructions.len());
        for block in 0..2 {
            let span = block * block_size..(block + 1) * block_size;
            let on_block = leading
                .iter()
                .filter(|targets| targets.iter().all(|q| span.contains(q)))
                .count();
            assert!(
                on_block >= 16,
                "{n}q block {block}: {on_block} leading Clifford gates"
            );
        }

        let result = sim::simulate(&circuit).seed(SEED).shots(64).unwrap();
        assert_eq!(
            result.metadata.backend,
            ResolvedBackend::Decomposed,
            "temporal_prefix_blocks/{n}q: the shots must run decomposed"
        );
    }
}

// Mirrors `random_clifford_circuit` in benches/circuits.rs: each brick CX lands
// with probability one half, so the register can split into blocks.
fn random_clifford_circuit(n: usize, depth: usize, seed: u64) -> prism_q::circuit::Circuit {
    use prism_q::Gate;
    use rand::{RngExt, SeedableRng};

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = prism_q::circuit::Circuit::new(n, 0);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];
    for layer in 0..depth {
        for q in 0..n {
            let gate = rng.random_range(0..cliffords.len());
            circuit.add_gate(cliffords[gate].clone(), &[q]);
        }
        for q in (layer % 2..n - 1).step_by(2) {
            if rng.random_bool(0.5) {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    circuit
}

// `factored_stabilizer/scaling` and `factored_stabilizer/local_blocks` price
// the backend that merges and splits clusters as the circuit runs, so their
// fixtures must split into several blocks and still run whole on one
// factored-stabilizer backend. The explicit tableau run is the witness that a
// fixture splits: that kind keeps the per-block route.
#[test]
fn factored_stabilizer_block_rows_run_whole() {
    let mut rows = Vec::new();
    for n in [10usize, 50, 100, 500, 1000] {
        rows.push((format!("scaling/{n}"), random_clifford_circuit(n, 10, SEED)));
    }
    for (blocks, size) in [(5usize, 2usize), (5, 4), (10, 5), (20, 5), (50, 5)] {
        rows.push((
            format!("local_blocks/{blocks}x{size}"),
            circuits::local_clifford_blocks(blocks, size, 10, SEED),
        ));
    }
    for (label, circuit) in rows {
        let split = sim::simulate(&circuit)
            .backend(BackendKind::Stabilizer)
            .seed(42)
            .run()
            .unwrap();
        assert!(
            split.metadata.blocks.len() > 1,
            "{label}: the fixture no longer splits, so the row prices one cluster"
        );
        let whole = sim::simulate(&circuit)
            .backend(BackendKind::FactoredStabilizer)
            .seed(42)
            .run()
            .unwrap();
        assert_eq!(
            whole.metadata.backend,
            ResolvedBackend::FactoredStabilizer,
            "{label}: ran per block rather than on one factored-stabilizer backend"
        );
    }
}

#[test]
fn clifford_register_counts_row_fills_the_register() {
    let mut circuit = circuits::clifford_t_circuit(40, 20, 0.05, SEED);
    circuit.measure_all();
    let counts = sim::simulate(&circuit)
        .seed(42)
        .sample_counts(1000)
        .unwrap();
    assert_eq!(counts.metadata.reason, RouteRule::CliffordRegister);
}

// The noisy dynamic rows each price one form of the rewrite to terminal measurements: an
// ancilla reused after its measurement, and a guard comparing several bits.
#[test]
fn noisy_dynamic_rows_take_their_route() {
    for (label, circuit, rule) in [
        (
            "rep_d9_active_reset_p001",
            bench_fixtures::active_reset_repetition_circuit(9, 9),
            RouteRule::CompiledCliffordSampler,
        ),
        (
            "rep_d5_lookup_p001",
            bench_fixtures::lookup_repetition_circuit(5, 5),
            RouteRule::CompiledCliffordSampler,
        ),
    ] {
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let plan = sim::simulate(&circuit)
            .noise(&noise)
            .plan(sim::Query::Shots(2_000))
            .unwrap();
        let result = sim::simulate(&circuit)
            .noise(&noise)
            .seed(SEED)
            .shots(64)
            .unwrap();
        assert_eq!(plan.rule, rule, "{label}: plan");
        assert_eq!(result.metadata.reason, rule, "{label}: run");
    }
}

fn price_layers(n: usize, t: usize) -> prism_q::circuit::Circuit {
    bench_fixtures::random_clifford_t_layers(n, 10, t, SEED)
}

fn price_scrambled(n: usize, t: usize) -> prism_q::circuit::Circuit {
    let mut circuit = bench_fixtures::scrambled_clifford_t_circuit(n, t, SEED);
    circuit.measure_all();
    circuit
}

fn price_distinct() -> prism_q::circuit::Circuit {
    bench_fixtures::distinct_t_layers(17, 10, 16, SEED)
}

fn price_onelayer() -> prism_q::circuit::Circuit {
    let mut circuit = bench_fixtures::random_t_insertion_circuit(20, 80, SEED);
    circuit.measure_all();
    circuit
}

/// The shot plan `kind` takes for `circuit`, without running it.
fn shot_plan(
    kind: BackendKind,
    circuit: &prism_q::circuit::Circuit,
    shots: usize,
) -> (ResolvedBackend, RouteRule) {
    let plan = sim::simulate(circuit)
        .backend(kind)
        .seed(42)
        .plan(sim::Query::Shots(shots))
        .unwrap();
    (plan.backend, plan.rule)
}

// `auto/clifford_t_price` prices one circuit on several engines, so each arm must land
// on the engine it names. The `statevector` rows must also stay one block, or the
// `auto` arm beside them would time the decomposed route. `auto/layers_28q_t24` runs
// only where `PRISM_BENCH_HIGH_QUBITS` admits 28 qubits, so its pin fixes the cap at 28
// rather than reading the host's.
#[test]
fn clifford_t_price_rows_pin_their_engines() {
    let register = (ResolvedBackend::StabilizerRank, RouteRule::CliffordRegister);
    let terminal_statevector = (ResolvedBackend::Statevector, RouteRule::TerminalStatevector);

    let mut register_rows = Vec::new();
    for (n, t) in [
        (26, 24),
        (28, 24),
        (30, 24),
        (32, 24),
        (26, 10),
        (26, 20),
        (20, 40),
    ] {
        register_rows.push((format!("layers_{n}q_t{t}"), price_layers(n, t)));
    }
    for (n, t) in [(26, 16), (26, 18), (26, 20), (20, 20), (22, 20), (32, 20)] {
        register_rows.push((format!("scrambled_{n}q_t{t}"), price_scrambled(n, t)));
    }
    register_rows.push(("onelayer_20q_t80".into(), price_onelayer()));
    register_rows.push(("distinct_17q_t16".into(), price_distinct()));
    for (label, circuit) in &register_rows {
        assert_eq!(
            shot_plan(BackendKind::StabilizerRank, circuit, 1024),
            register,
            "register/{label}"
        );
    }
    // Past 20 T gates the register width is not bounded by the T count, and a row that
    // outgrew the cap would fail rather than time.
    for (label, circuit) in register_rows.iter().filter(|(l, _)| l.ends_with("_t24")) {
        sim::simulate(circuit)
            .backend(BackendKind::StabilizerRank)
            .seed(42)
            .shots(1)
            .unwrap_or_else(|e| panic!("register/{label}: {e}"));
    }

    let mut statevector_rows = Vec::new();
    for n in [22, 24, 26, 28] {
        statevector_rows.push((format!("layers_{n}q_t24"), price_layers(n, 24)));
    }
    for n in [20, 22, 24] {
        statevector_rows.push((format!("scrambled_{n}q_t20"), price_scrambled(n, 20)));
    }
    statevector_rows.push(("layers_20q_t40".into(), price_layers(20, 40)));
    statevector_rows.push(("distinct_17q_t16".into(), price_distinct()));
    for (label, circuit) in &statevector_rows {
        assert_eq!(
            circuit.independent_subsystems().len(),
            1,
            "statevector/{label}: the register splits"
        );
        assert_eq!(
            shot_plan(BackendKind::Statevector, circuit, 1024),
            terminal_statevector,
            "statevector/{label}"
        );
    }

    let auto_rows = [
        ("layers_22q_t24", price_layers(22, 24)),
        ("layers_24q_t24", price_layers(24, 24)),
        ("layers_26q_t24", price_layers(26, 24)),
        ("scrambled_22q_t20", price_scrambled(22, 20)),
        ("layers_20q_t40", price_layers(20, 40)),
    ];
    for (label, circuit) in &auto_rows {
        assert_eq!(
            shot_plan(BackendKind::Auto, circuit, 1024),
            register,
            "auto/{label}"
        );
    }
    // The bracket on the statevector side of the register's cost constant, and the
    // single-layer row the factored route takes ahead of the register.
    assert_eq!(
        shot_plan(BackendKind::Auto, &price_distinct(), 1024),
        terminal_statevector,
        "auto/distinct_17q_t16"
    );
    let factored = (ResolvedBackend::Factored, RouteRule::NativeSampling);
    for kind in [BackendKind::Factored, BackendKind::Auto] {
        assert_eq!(
            shot_plan(kind.clone(), &price_onelayer(), 1024),
            factored,
            "{kind:?}/onelayer_20q_t80"
        );
    }
    let wide = price_layers(28, 24);
    let plan = sim::simulate(&wide)
        .limits(prism_q::Limits::new().max_statevector_qubits(28))
        .seed(42)
        .plan(sim::Query::Shots(1024))
        .unwrap();
    assert_eq!((plan.backend, plan.rule), register, "auto/layers_28q_t24");

    for (t, shots) in [(6, 16), (8, 4)] {
        for n in [26, 32] {
            let circuit = bench_fixtures::stabilizer_rank_mid_circuit_shot_circuit(n, t, SEED);
            assert!(!circuit.has_terminal_measurements_only());
            assert_eq!(
                shot_plan(BackendKind::StabilizerRank, &circuit, shots),
                register,
                "rank_mid/mid_{n}q_t{t}_{shots}shots"
            );
        }
    }
    for n in [22, 24, 26] {
        let circuit = bench_fixtures::clifford_t_layers_mid_circuit(n, 10, 8, SEED);
        assert_eq!(
            shot_plan(BackendKind::Statevector, &circuit, 64),
            (ResolvedBackend::Statevector, RouteRule::PerShot),
            "statevector_mid/layers_{n}q_t8"
        );
    }

    // The deferral moves the layered and `mid` tails to the end, and the register takes
    // them. A guarded `S` keeps its measurement in place: the tree takes it under the cap,
    // the branches past it up to 6 T, and the tree beyond. The fixture's chain bonds drop
    // with probability one half, so the tree splits it into blocks or factors it. The 32-qubit pins fix the
    // cap at 30, below 32 as on the bench host, rather than reading this host's.
    let factored = (ResolvedBackend::Factored, RouteRule::PerShot);
    let blocks = (ResolvedBackend::Decomposed, RouteRule::PerShot);
    let past_cap = |circuit: &prism_q::circuit::Circuit, shots| {
        let plan = sim::simulate(circuit)
            .limits(prism_q::Limits::new().max_statevector_qubits(30))
            .seed(42)
            .plan(sim::Query::Shots(shots))
            .unwrap();
        (plan.backend, plan.rule)
    };
    for (label, circuit, shots) in auto_mid_rows() {
        let route = if label.starts_with("guarded_s_32q") {
            past_cap(&circuit, shots)
        } else {
            shot_plan(BackendKind::Auto, &circuit, shots)
        };
        let expected = match label.as_str() {
            "guarded_s_24q_t6" => factored,
            "guarded_s_24q_t8" | "guarded_s_32q_t8" => blocks,
            _ => register,
        };
        assert_eq!(route, expected, "auto_mid/{label}");
    }
}

// `auto/clifford_t_expectation` times the product-gauge register on every row. The cap is
// fixed at 29, as on the bench host, so the 30, 32 and 64 qubit rows sit above it.
#[test]
fn clifford_t_expectation_rows_take_the_register() {
    let register = (ResolvedBackend::StabilizerRank, RouteRule::CliffordRegister);
    let plan = |circuit: &prism_q::circuit::Circuit, query: sim::Query<'_>| {
        let plan = sim::simulate(circuit)
            .limits(prism_q::Limits::new().max_statevector_qubits(29))
            .seed(42)
            .plan(query)
            .unwrap();
        (plan.backend, plan.rule)
    };
    let rows = [
        ("layers_30q_t40", price_layers(30, 40)),
        ("layers_32q_t40", price_layers(32, 40)),
        (
            "local_layers_64q_t40",
            bench_fixtures::local_clifford_t_layers(64, 10, 40, SEED),
        ),
        ("layers_24q_t24", price_layers(24, 24)),
    ];
    for (label, circuit) in &rows {
        let circuit = circuit.without_measurements();
        let observables = bench_fixtures::clifford_t_expectation_observables(circuit.num_qubits);
        assert_eq!(
            plan(&circuit, sim::Query::Expectation(&observables)),
            register,
            "expectation/{label}"
        );
    }
    let marginals = price_layers(24, 24).without_measurements();
    assert_eq!(
        plan(&marginals, sim::Query::Marginals),
        register,
        "marginals/layers_24q_t24"
    );
}

fn auto_mid_rows() -> Vec<(String, prism_q::circuit::Circuit, usize)> {
    let mut rows = Vec::new();
    for n in [22, 24, 26] {
        let circuit = bench_fixtures::clifford_t_layers_mid_circuit(n, 10, 8, SEED);
        rows.push((format!("layers_{n}q_t8"), circuit, 64));
    }
    for n in [26, 32] {
        let circuit = bench_fixtures::stabilizer_rank_mid_circuit_shot_circuit(n, 6, SEED);
        rows.push((format!("mid_{n}q_t6_16shots"), circuit, 16));
    }
    for n in [24, 32] {
        for t in [6, 8] {
            let circuit = bench_fixtures::guarded_s_mid_circuit(n, t, SEED);
            rows.push((format!("guarded_s_{n}q_t{t}"), circuit, 4));
        }
    }
    rows
}

#[test]
fn local_clifford_t_layers_pairs_neighbours() {
    use prism_q::{Gate, Instruction};

    let circuit = bench_fixtures::local_clifford_t_layers(9, 4, 6, SEED);
    let mut cx = 0;
    for inst in &circuit.instructions {
        if let Instruction::Gate {
            gate: Gate::Cx,
            targets,
        } = inst
        {
            assert_eq!(targets[1], targets[0] + 1);
            cx += 1;
        }
    }
    assert_eq!(cx, 16);
    assert_eq!(circuit.t_count(), 6);
}

// Deterministic counts behind `auto/clifford_t_price`, printed rather than asserted:
// the register's width, rotation work, reduction passes and parts in the product gauge
// per terminal fixture, the route `Auto` takes, and whether a width-priced rule would
// need a tableau dry run;
// per mid-circuit fixture the branch count and the inner products one shot costs.
// Run with `--features "parallel bench-internal" --no-capture`.
#[cfg(feature = "bench-internal")]
#[test]
fn clifford_t_price_census() {
    use prism_q::{Gate, Instruction};

    const REGISTER_PASS_STATEVECTOR_AMPLITUDES: f64 = 25.0;
    const MAX_REGISTER_QUBITS: usize = 20;

    let mut terminal = Vec::new();
    for (n, t) in [
        (20, 40),
        (22, 24),
        (24, 24),
        (26, 10),
        (26, 20),
        (26, 24),
        (28, 24),
        (30, 24),
        (32, 24),
        (26, 40),
        (30, 40),
        (32, 40),
    ] {
        terminal.push((format!("layers_{n}q_t{t}"), price_layers(n, t)));
    }
    terminal.push((
        "local_layers_64q_t40".into(),
        bench_fixtures::local_clifford_t_layers(64, 10, 40, SEED),
    ));
    for n in [16, 20, 24] {
        terminal.push((
            format!("clifford_t_shots/{n}q_t10"),
            bench_fixtures::random_clifford_t_layers(n, 10, 10, SEED),
        ));
    }
    for (n, t) in [
        (20, 20),
        (22, 20),
        (24, 20),
        (26, 16),
        (26, 18),
        (26, 20),
        (32, 20),
    ] {
        terminal.push((format!("scrambled_{n}q_t{t}"), price_scrambled(n, t)));
    }
    terminal.push(("onelayer_20q_t80".into(), price_onelayer()));
    terminal.push(("distinct_17q_t16".into(), price_distinct()));
    let mut counts_row = circuits::clifford_t_circuit(40, 20, 0.05, SEED);
    counts_row.measure_all();
    terminal.push(("clifford_register_counts/40q_d20_t5pct".into(), counts_row));

    println!(
        "{:<40} {:>3} {:>3} {:>3} {:>4} {:>8} {:>6} {:>3} {:>7} {:>5} {:>4} {:>5} {:>5} {:>6}  auto route",
        "fixture",
        "n",
        "t",
        "k",
        "e",
        "R/2^k",
        "passes",
        "r",
        "dry_run",
        "parts",
        "kmax",
        "dense",
        "split",
        "pairs"
    );
    let mut split_lines = Vec::new();
    for (label, circuit) in &terminal {
        let census = prism_q::clifford_register_census(circuit).unwrap();
        let (n, t, k) = (circuit.num_qubits, census.t_count, census.width);
        let m = circuit.measurement_map().len();
        let instructions = circuit.instructions.len() as f64;
        let w_alt: f64 = circuit
            .independent_subsystems()
            .iter()
            .map(|block| instructions * 2f64.powi(block.len() as i32))
            .sum();
        let w_reg = |k: usize, rotation_amps: f64| {
            let reduction = 3.0 * (k.min(m) + 1) as f64 * 2f64.powi(k as i32);
            REGISTER_PASS_STATEVECTOR_AMPLITUDES * (rotation_amps + reduction)
        };
        let kb = t.min(n);
        let bound_admits = kb <= MAX_REGISTER_QUBITS
            && w_reg(kb, (t - kb + 2) as f64 * 2f64.powi(kb as i32)) < w_alt;
        let k_stop = (0..=kb.min(MAX_REGISTER_QUBITS))
            .rev()
            .find(|&k| w_reg(k, 2.0 * 2f64.powi(k as i32)) < w_alt);
        let dry_run = !bound_admits && k_stop.is_some();
        let (backend, rule) = shot_plan(BackendKind::Auto, circuit, 1024);
        let show = |v: Option<usize>| v.map_or("-".to_string(), |v| v.to_string());
        // log2 of the rotation work `(e + 2) 2^k`, for the one register and summed over
        // its parts in the product gauge.
        let dense = ((t - k + 2) as f64).log2() + k as f64;
        let split = census
            .parts
            .iter()
            .map(|p| (p.excess + 2) as f64 * 2f64.powi(p.width as i32))
            .sum::<f64>()
            .log2();
        let kmax = census.parts.first().map_or(0, |p| p.width);
        println!(
            "{:<40} {:>3} {:>3} {:>3} {:>4} {:>8.2} {:>6} {:>3} {:>7} {:>5} {:>4} {:>5.1} {:>5.1} {:>6}  {backend:?}/{rule:?}",
            label,
            n,
            t,
            k,
            t as i64 - k as i64,
            census.rotation_work as f64 / 2f64.powi(k as i32),
            show(census.reduction_passes),
            show(census.register_rows),
            dry_run,
            census.parts.len(),
            kmax,
            dense,
            split,
            census.row_pair_checks,
        );
        let widest: Vec<String> = census
            .parts
            .iter()
            .filter(|p| p.width > 1 || p.excess > 0)
            .take(6)
            .map(|p| format!("{}w{}e{}r", p.width, p.excess, p.rows))
            .collect();
        let singles = census
            .parts
            .iter()
            .filter(|p| p.width == 1 && p.excess == 0)
            .count();
        split_lines.push(format!(
            "{label:<40} {} + {singles} single",
            widest.join(" ")
        ));
    }
    println!();
    println!(
        "{:<40} parts past one qubit (width, excess, table bits)",
        "fixture"
    );
    for line in &split_lines {
        println!("{line}");
    }

    let mut mid = Vec::new();
    for (t, shots) in [(6, 16), (8, 4)] {
        for n in [26, 32] {
            let circuit = bench_fixtures::stabilizer_rank_mid_circuit_shot_circuit(n, t, SEED);
            mid.push((
                format!("rank_mid/mid_{n}q_t{t}_{shots}shots"),
                circuit,
                shots,
            ));
        }
    }
    for n in [22, 24, 26] {
        let circuit = bench_fixtures::clifford_t_layers_mid_circuit(n, 10, 8, SEED);
        mid.push((format!("statevector_mid/layers_{n}q_t8"), circuit, 64));
    }
    for (label, circuit, shots) in auto_mid_rows() {
        mid.push((format!("auto_mid/{label}"), circuit, shots));
    }
    println!();
    println!(
        "{:<40} {:>3} {:>3} {:>6} {:>6} {:>12}  auto route",
        "mid fixture", "n", "t", "B", "events", "inner_prods"
    );
    for (label, circuit, shots) in &mid {
        let t = circuit
            .instructions
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    Instruction::Gate {
                        gate: Gate::T | Gate::Tdg,
                        ..
                    }
                )
            })
            .count();
        let events = circuit
            .instructions
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    Instruction::Measure { .. } | Instruction::Reset { .. }
                )
            })
            .count();
        let branches = 1u64 << t;
        // Each event takes the norm of both projections, `B (B + 1) / 2` overlaps each
        // while no branch projects to zero.
        let inner_products = events as u64 * branches * (branches + 1);
        let (backend, rule) = shot_plan(BackendKind::Auto, circuit, *shots);
        println!(
            "{:<40} {:>3} {:>3} {:>6} {:>6} {:>12}  {backend:?}/{rule:?}",
            label, circuit.num_qubits, t, branches, events, inner_products
        );
    }
}
