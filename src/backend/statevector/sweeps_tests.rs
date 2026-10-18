use super::*;
use crate::circuits;

const SEED: u64 = 0xDEAD_BEEF;

fn take_ticks() -> usize {
    PASS_TICKS.with(|ticks| ticks.replace(0))
}

fn trotter(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for (coefficient, factors) in &circuits::jordan_wigner_hamiltonian(n, 200, SEED) {
        if !factors.is_empty() {
            c.add_pauli_rotation(0.1 * coefficient, factors);
        }
    }
    c
}

fn measured(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 2);
    c.instructions = circuits::hardware_efficient_ansatz(n, 2, SEED).instructions;
    c.add_measure(3, 0);
    c.add_reset(n - 1);
    c.add_gate(Gate::H, &[n - 1]);
    c.add_measure(n - 1, 1);
    c
}

fn corpus(n: usize) -> Vec<(&'static str, Circuit)> {
    vec![
        ("hea_l5", circuits::hardware_efficient_ansatz(n, 5, SEED)),
        ("random_d10", circuits::random_circuit(n, 10, SEED)),
        ("qaoa_l3", circuits::qaoa_circuit(n, 3, SEED)),
        ("qv", circuits::quantum_volume_circuit(n, n, SEED)),
        ("qft_textbook", circuits::qft_circuit(n)),
        (
            "diag_mixed_l6",
            circuits::diagonal_mixed_circuit(n, 6, SEED),
        ),
        ("trotter", trotter(n)),
        ("measured", measured(n)),
    ]
}

fn pool_width() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

#[test]
fn sweep_profile_matches_kernel_pass_ticks() {
    for n in [14, 16, 18, 20] {
        for (name, circuit) in corpus(n) {
            let mut backend = StatevectorBackend::new(42);
            let expanded = crate::sim::expand_for_backend(&backend, &circuit);
            let fused = crate::sim::fuse_for_backend(&backend, &expanded);
            backend.init(n, circuit.num_classical_bits).unwrap();
            let threads = pool_width();
            take_ticks();
            for (i, inst) in fused.instructions.iter().enumerate() {
                backend.apply(inst).unwrap();
                let ticked = take_ticks();
                let modeled = sweep_profile(std::slice::from_ref(inst), n, threads);
                let kind = SweepKind::ALL
                    .into_iter()
                    .find(|&kind| modeled.count(kind).instructions > 0)
                    .map_or("barrier", SweepKind::name);
                assert_eq!(
                    ticked,
                    modeled.passes(),
                    "{name}/{n} instruction {i} ({kind})"
                );
            }
            let profile = sweep_profile_for(&circuit, threads);
            assert_eq!(profile, sweep_profile(&fused.instructions, n, threads));
            let by_kind = SweepKind::ALL.map(|kind| profile.count(kind));
            assert_eq!(
                by_kind.iter().map(|c| c.tile_apps).sum::<usize>(),
                profile.tile_apps()
            );
        }
    }
}

#[test]
fn sweep_lower_bound_counts_cover_and_chains() {
    let n = 40;
    let low = multi_2q_low_bits();
    let budget = multi_2q_high_budget_for(n);
    let qubits: Vec<usize> = (low..low + 2 * budget).collect();
    let gate = |gate: Gate, targets: &[usize]| Instruction::Gate {
        gate,
        targets: targets.iter().copied().collect(),
    };

    let independent: Vec<Instruction> = qubits.iter().map(|&q| gate(Gate::H, &[q])).collect();
    assert_eq!(lower_bound(&independent, n, 1), 2);

    let swaps: Vec<Instruction> = qubits
        .windows(2)
        .map(|pair| gate(Gate::Swap, pair))
        .collect();
    assert_eq!(lower_bound(&swaps, n, 1), 3);

    let mut diagonal: Vec<Instruction> =
        qubits.windows(2).map(|pair| gate(Gate::Cz, pair)).collect();
    diagonal.push(gate(Gate::Cx, &[qubits[0], low - 1]));
    assert_eq!(lower_bound(&diagonal, n, 1), 1);

    let mut split = independent[..2].to_vec();
    split.push(Instruction::Measure {
        qubit: qubits[0],
        classical_bit: 0,
    });
    split.extend(independent[2..4].iter().cloned());
    assert_eq!(lower_bound(&split, n, 1), 4);
}

#[test]
fn sweep_lower_bound_stays_under_the_profiled_passes() {
    for (name, circuit) in corpus(20) {
        let bound = sweep_lower_bound(&circuit, pool_width());
        let passes = sweep_profile_for(&circuit, pool_width()).passes();
        assert!(
            bound >= 1 && bound <= passes,
            "{name}: bound {bound}, passes {passes}"
        );
    }
}
