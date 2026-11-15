use super::*;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const CLIFFORDS_1Q: [Gate; 8] = [
    Gate::H,
    Gate::S,
    Gate::Sdg,
    Gate::SX,
    Gate::SXdg,
    Gate::X,
    Gate::Y,
    Gate::Z,
];

fn random_clifford_t(n: usize, gates: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, n);
    let mut t_left = t_count;
    for g in 0..gates {
        let slots_left = gates - g;
        if t_left > 0 && rng.random_range(0..slots_left) < t_left {
            let gate = if rng.random_bool(0.5) {
                Gate::T
            } else {
                Gate::Tdg
            };
            c.add_gate(gate, &[rng.random_range(0..n)]);
            t_left -= 1;
        } else if n > 1 && rng.random_bool(0.4) {
            let a = rng.random_range(0..n);
            let b = (a + rng.random_range(1..n)) % n;
            let gate = [Gate::Cx, Gate::Cz, Gate::Swap][rng.random_range(0..3)].clone();
            c.add_gate(gate, &[a, b]);
        } else {
            let gate = CLIFFORDS_1Q[rng.random_range(0..CLIFFORDS_1Q.len())].clone();
            c.add_gate(gate, &[rng.random_range(0..n)]);
        }
    }
    c
}

fn statevector_probabilities(circuit: &Circuit) -> Vec<f64> {
    let mut sv = StatevectorBackend::new(42);
    sv.init(circuit.num_qubits, 0).unwrap();
    for inst in &circuit.instructions {
        if matches!(inst, Instruction::Gate { .. }) {
            sv.apply(inst).unwrap();
        }
    }
    sv.probabilities().unwrap()
}

fn assert_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-10,
            "{label}: outcome {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn register_distribution_matches_the_statevector() {
    for seed in 0..60u64 {
        let n = 1 + (seed as usize % 6);
        let t_count = (seed as usize / 6) % 7;
        let mut circuit = random_clifford_t(n, 12 + 4 * n, t_count, seed);
        let want = statevector_probabilities(&circuit);
        for q in 0..n {
            circuit.add_measure(q, q);
        }
        let sampler = TerminalSampler::compile_every_qubit(&circuit, MAX_REGISTER_QUBITS)
            .unwrap()
            .expect("the register fits");
        let label = format!("seed {seed}, n {n}, t {t_count}");
        assert_close(&sampler.distribution(), &want, &label);
    }
}

#[test]
fn register_gates_match_their_basis_action() {
    let width = 5;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let start: Vec<Complex64> = (0..1usize << width)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let mut gates = Vec::new();
    for m in 0..width {
        gates.push(RegisterGate::H(m));
        gates.push(RegisterGate::Sdg(m));
        let others = ((1usize << width) - 1) & !(1 << m);
        let mut fans: Vec<usize> = (0..width).filter(|&c| c != m).map(|c| 1 << c).collect();
        fans.push(others);
        fans.extend((0..3).map(|_| others & rng.random_range(1..1usize << width)));
        for controls in fans.into_iter().filter(|&c| c != 0) {
            gates.push(RegisterGate::CxFan {
                controls,
                target: m,
            });
        }
    }
    for gate in &gates {
        let mut want = start.clone();
        match *gate {
            RegisterGate::H(m) => {
                for b in (0..want.len()).filter(|b| (b >> m) & 1 == 0) {
                    let (a0, a1) = (start[b], start[b | 1 << m]);
                    want[b] = (a0 + a1) * FRAC_1_SQRT_2;
                    want[b | 1 << m] = (a0 - a1) * FRAC_1_SQRT_2;
                }
            }
            RegisterGate::Sdg(m) => {
                for b in (0..want.len()).filter(|b| (b >> m) & 1 == 1) {
                    want[b] = start[b] * Complex64::new(0.0, -1.0);
                }
            }
            RegisterGate::CxFan { controls, target } => {
                for b in (0..want.len()).filter(|b| (b & controls).count_ones() % 2 == 1) {
                    want[b] = start[b ^ 1 << target];
                }
            }
        }
        let mut got = start.clone();
        apply_to_amps(&mut got, gate);
        assert_eq!(got, want);
    }
}

#[test]
fn every_qubit_distribution_ignores_the_circuit_measurements() {
    let n = 5;
    let mut circuit = random_clifford_t(n, 40, 8, 42);
    let want = statevector_probabilities(&circuit);
    circuit.add_measure(3, 0);
    let sampler = TerminalSampler::compile_every_qubit(&circuit, MAX_REGISTER_QUBITS)
        .unwrap()
        .unwrap();
    assert!(sampler.register_width() <= n);
    assert_close(&sampler.distribution(), &want, "every qubit");
}

#[test]
fn register_declines_past_its_width() {
    let mut circuit = Circuit::new(4, 4);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
        circuit.add_measure(q, q);
    }
    assert!(
        TerminalSampler::compile_every_qubit(&circuit, 3)
            .unwrap()
            .is_none()
    );
    assert!(
        TerminalSampler::compile_every_qubit(&circuit, 4)
            .unwrap()
            .is_some()
    );
}

#[test]
fn register_width_dry_run_agrees_with_compile() {
    for seed in 0..60u64 {
        let n = 2 + (seed as usize % 6);
        let t_count = 1 + (seed as usize / 6) % 8;
        let mut circuit = random_clifford_t(n, 12 + 4 * n, t_count, seed);
        for q in 0..n {
            circuit.add_measure(q, q);
        }
        for max_width in 0..=n {
            assert_eq!(
                register_price(&circuit, max_width)
                    .unwrap()
                    .map(|(width, _)| width),
                TerminalSampler::compile_every_qubit(&circuit, max_width)
                    .unwrap()
                    .map(|sampler| sampler.register_width()),
                "seed {seed}, n {n}, t {t_count}, max width {max_width}"
            );
        }
    }
}

#[test]
fn auto_shots_take_the_register_from_16_qubits() {
    let ladder = |n: usize, t_count: usize| {
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..t_count {
            c.add_gate(Gate::T, &[q % n]);
            c.add_gate(Gate::H, &[q % n]);
        }
        for q in 0..n {
            c.add_measure(q, q);
        }
        c
    };
    let route = |c: &Circuit| {
        crate::simulate(c)
            .seed(1)
            .shots(16)
            .unwrap()
            .metadata
            .backend
    };
    use crate::sim::ResolvedBackend;
    assert_eq!(route(&ladder(16, 10)), ResolvedBackend::StabilizerRank);
    assert_eq!(route(&ladder(12, 4)), ResolvedBackend::Statevector);
    // The split holds the 16 anchors in parts far narrower than the circuit.
    assert_eq!(route(&ladder(16, 16)), ResolvedBackend::StabilizerRank);
}

#[test]
fn cut_keeps_strictly_larger_amplitudes_before_ties_in_index_order() {
    let mut state = CliffordRegister::new(2);
    state.width = 2;
    state.amps = [0.5, 0.5, 0.5, 0.6]
        .iter()
        .map(|&a| Complex64::new(a, 0.0))
        .collect();
    let mut prune = Pruning::new(2);
    state.cut(&mut prune);
    let kept: Vec<f64> = state.amps.iter().map(|a| a.re).collect();
    assert_eq!(kept, [0.5, 0.0, 0.0, 0.6]);
    assert_eq!(prune.pruned, 2);
    assert!((prune.discarded - 0.5f64.sqrt()).abs() < 1e-15);

    let mut sparse = CliffordRegister::new(2);
    sparse.width = 2;
    sparse.amps = vec![
        Complex64::new(0.6, 0.0),
        Complex64::new(0.0, 0.8),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let before = sparse.amps.clone();
    let mut prune = Pruning::new(2);
    sparse.cut(&mut prune);
    assert_eq!(sparse.amps, before);
    assert_eq!((prune.pruned, prune.discarded), (0, 0.0));
}

#[test]
fn table_fold_matches_the_bit_gather_exactly() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let swapped = |width: usize, a: usize, b: usize| {
        let mut p: Vec<usize> = (0..width).collect();
        p.swap(a, b);
        p
    };
    let shapes: Vec<(usize, Vec<usize>)> = vec![
        (0, vec![]),
        (5, (0..5).collect()),
        (8, (0..8).collect()),
        (12, (0..12).collect()),
        (5, vec![0, 1, 2, 4, 3]),
        (12, swapped(12, 9, 11)),
        (12, swapped(12, 0, 7)),
        (12, vec![11, 3, 0, 8, 5, 1, 10, 2, 7, 4, 9, 6]),
        (12, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]),
        (12, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]),
        (12, vec![1, 0, 2, 3, 5, 6, 7, 8, 9, 10, 11]),
        (12, vec![9, 2, 5]),
        (6, vec![2, 0, 5]),
    ];
    for (width, pivots) in shapes {
        let amps: Vec<Complex64> = (0..1usize << width)
            .map(|b| {
                if b % 7 == 3 {
                    Complex64::new(0.0, 0.0)
                } else {
                    let scale = 10f64.powi(rng.random_range(-6..1));
                    Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5) * scale
                }
            })
            .collect();
        let mut want = vec![0.0f64; 1 << pivots.len()];
        for (b, amp) in amps.iter().enumerate() {
            let index = pivots
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &c)| acc | (((b >> c) & 1) << i));
            want[index] += amp.norm_sqr();
        }
        let got = fold_table(&amps, &pivots, width);
        assert!(
            got.iter()
                .map(|p| p.to_bits())
                .eq(want.iter().map(|p| p.to_bits())),
            "width {width}, pivots {pivots:?}"
        );
    }
}

fn register_pair(circuit: &Circuit, budget: usize) -> (Vec<Complex64>, Vec<Complex64>, Pruning) {
    let exact = build_register(circuit, MAX_REGISTER_QUBITS, true, None, &mut Vec::new())
        .unwrap()
        .unwrap();
    let mut prune = Pruning::new(budget);
    let pruned = build_register(
        circuit,
        MAX_REGISTER_QUBITS,
        true,
        Some(&mut prune),
        &mut Vec::new(),
    )
    .unwrap()
    .unwrap();
    (exact.amps, pruned.amps, prune)
}

#[test]
fn one_cut_drops_exactly_the_reported_norm() {
    let mut circuit = Circuit::new(3, 0);
    for q in 0..3 {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
        circuit.add_gate(Gate::H, &[q]);
    }
    let (phi, psi, prune) = register_pair(&circuit, 4);
    assert_eq!(prune.pruned, 4);
    let dropped: f64 = phi.iter().zip(&psi).map(|(a, b)| (a - b).norm_sqr()).sum();
    let kept: f64 = psi.iter().map(|a| a.norm_sqr()).sum();
    let d2 = prune.discarded * prune.discarded;
    assert!((dropped - d2).abs() < 1e-14, "{dropped} vs {d2}");
    assert!((1.0 - kept - d2).abs() < 1e-14, "{kept} vs {d2}");
}

#[test]
fn pruned_register_stays_within_the_summed_cut_norms() {
    let mut multi_cut_runs = 0;
    for seed in 0..12u64 {
        let circuit = random_clifford_t(6, 60, 14, seed);
        for budget in [4, 12, 24] {
            let (phi, psi, prune) = register_pair(&circuit, budget);
            assert!(psi.iter().filter(|a| a.norm_sqr() != 0.0).count() <= budget);
            let distance = phi
                .iter()
                .zip(&psi)
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let n2: f64 = psi.iter().map(|a| a.norm_sqr()).sum();
            let overlap: Complex64 = phi.iter().zip(&psi).map(|(a, b)| a.conj() * b).sum();
            let fidelity = overlap.norm_sqr() / n2;
            let floor = crate::StabRankResult {
                probabilities: Vec::new(),
                num_terms: 0,
                t_count: 0,
                pruned_count: prune.pruned,
                discarded_weight: prune.discarded,
                kept_norm_sqr: n2,
            }
            .fidelity_bound();
            let label = format!("seed {seed}, budget {budget}");
            assert!(distance <= prune.discarded + 1e-12, "{label}: {distance}");
            assert!(1.0 - n2 <= prune.discarded.powi(2) + 1e-12, "{label}");
            assert!(fidelity >= floor - 1e-12, "{label}: {fidelity} < {floor}");
            if prune.pruned == 0 {
                assert_eq!(prune.discarded, 0.0, "{label}");
                assert_eq!(psi, phi, "{label}");
            } else if distance < prune.discarded - 1e-9 {
                multi_cut_runs += 1;
            }
        }
    }
    assert!(multi_cut_runs > 0);
}
