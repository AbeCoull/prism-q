use super::*;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use rand::RngExt;

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

/// Enumerate every coin assignment and register outcome the sampler can draw.
fn exact_distribution(sampler: &TerminalSampler) -> Vec<f64> {
    let coins = sampler
        .outcome
        .iter()
        .filter(|o| matches!(o, RowOutcome::Random))
        .count();
    let mut dist = vec![0.0; 1 << sampler.num_classical_bits];
    let mut below = 0.0;
    for (index, &upto) in sampler.cumulative.iter().enumerate() {
        let p = (upto - below) / (1u64 << coins) as f64;
        below = upto;
        for assignment in 0..1usize << coins {
            let mut k = 0;
            let bits = sampler.classical_bits(index, || {
                k += 1;
                (assignment >> (k - 1)) & 1 == 1
            });
            let key = bits
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &b)| acc | (usize::from(b) << i));
            dist[key] += p;
        }
    }
    dist
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
        let sampler = TerminalSampler::compile(&circuit, MAX_REGISTER_QUBITS)
            .unwrap()
            .expect("the register fits");
        let label = format!("seed {seed}, n {n}, t {t_count}");
        assert_close(&exact_distribution(&sampler), &want, &label);
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
fn register_marginals_follow_the_measurement_map() {
    let n = 5;
    let base = random_clifford_t(n, 40, 5, 7);
    let full = statevector_probabilities(&base);
    let mut circuit = Circuit::new(n, 3);
    circuit.instructions = base.instructions.clone();
    circuit.add_measure(3, 0);
    circuit.add_measure(1, 2);
    circuit.add_measure(3, 1);

    let mut want = vec![0.0; 8];
    for (state, p) in full.iter().enumerate() {
        let (q3, q1) = ((state >> 3) & 1, (state >> 1) & 1);
        want[q3 | (q3 << 1) | (q1 << 2)] += p;
    }
    let sampler = TerminalSampler::compile(&circuit, MAX_REGISTER_QUBITS)
        .unwrap()
        .unwrap();
    assert_close(&exact_distribution(&sampler), &want, "subset");
}

#[test]
fn register_declines_past_its_width() {
    let mut circuit = Circuit::new(4, 4);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
        circuit.add_measure(q, q);
    }
    assert!(TerminalSampler::compile(&circuit, 3).unwrap().is_none());
    assert!(TerminalSampler::compile(&circuit, 4).unwrap().is_some());
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
                register_fits(&circuit, max_width).unwrap(),
                TerminalSampler::compile(&circuit, max_width)
                    .unwrap()
                    .is_some(),
                "seed {seed}, n {n}, t {t_count}, max width {max_width}"
            );
        }
    }
}

#[test]
fn register_samples_a_wide_register_from_one_rotated_qubit() {
    let n = 200;
    let mut circuit = Circuit::new(n, n);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    let sampler = TerminalSampler::compile(&circuit, MAX_REGISTER_QUBITS)
        .unwrap()
        .unwrap();
    let shots = sampler.sample(20_000, 42);
    assert!(shots.iter().all(|s| s.iter().all(|&b| b == s[0])));
    let zeros = shots.iter().filter(|s| !s[0]).count() as f64 / shots.len() as f64;
    let want = FRAC_PI_8.cos().powi(2);
    assert!(
        (zeros - want).abs() < 0.01,
        "P(0...0) = {zeros}, want {want}"
    );
}

#[test]
fn auto_shots_take_the_register_from_16_qubits_with_fewer_t_than_qubits() {
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
    assert_eq!(route(&ladder(16, 16)), ResolvedBackend::Statevector);
}
