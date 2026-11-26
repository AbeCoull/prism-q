use super::*;

fn pauli(letters: &str) -> RegisterPauli {
    let mut p = RegisterPauli {
        x: 0,
        z: 0,
        phase4: 0,
    };
    for (b, c) in letters.chars().enumerate() {
        match c {
            'X' => p.x |= 1 << b,
            'Z' => p.z |= 1 << b,
            'Y' => {
                p.x |= 1 << b;
                p.z |= 1 << b;
            }
            _ => {}
        }
    }
    p
}

fn parts_of(rows: &[&str], units: &[u64]) -> Vec<usize> {
    let basis = row_basis(rows.iter().map(|r| pauli(r)));
    let forms = unit_forms(units, &basis);
    canonical_parts(units, &forms)
        .iter()
        .map(|p| p.count_ones() as usize)
        .collect()
}

#[test]
fn canonical_parts_follow_the_row_space_in_every_order() {
    let singles = [1, 2, 4, 8];
    assert_eq!(parts_of(&["XXXX", "ZZZZ"], &singles), [4]);
    let rows = ["XXXX", "ZZZZ", "XXII", "ZZII"];
    let mut orders = 0;
    for a in 0..4 {
        for b in (0..4).filter(|&b| b != a) {
            for c in (0..4).filter(|&c| c != a && c != b) {
                let d = 6 - a - b - c;
                let ordered = [rows[a], rows[b], rows[c], rows[d]];
                assert_eq!(parts_of(&ordered, &singles), [2, 2], "{ordered:?}");
                orders += 1;
            }
        }
    }
    assert_eq!(orders, 24);
}

use crate::circuit::Instruction;
use crate::gates::Gate;
use crate::test_support::{assert_close, gate_only_statevector_probabilities};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The tier limits of the test plan, low enough that circuits of 10 to 14 qubits reach
/// every tier.
const TIER_LIMITS: SplitLimits = SplitLimits {
    pack_bits: 2,
    whole_width: 4,
    whole_first_width: 6,
    table_width: 6,
    side_width: 4,
    cheap_pairs: 1,
    max_pairs: 3,
    max_anchors: 10,
    total_log2: 7,
};

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

/// Layers of random one-qubit Cliffords with a T or Tdg on some qubits, then CX on random
/// pairs: `pairs` per layer, so few pairs leave the register in small parts.
fn layered_clifford_t(n: usize, depth: usize, t_count: usize, pairs: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, n);
    let mut t_left = t_count;
    for layer in 0..depth {
        for q in 0..n {
            c.add_gate(CLIFFORDS_1Q[rng.random_range(0..8)].clone(), &[q]);
            let slots = (depth - layer) * n - q;
            if t_left > 0 && rng.random_range(0..slots) < t_left {
                let gate = if rng.random_bool(0.5) {
                    Gate::T
                } else {
                    Gate::Tdg
                };
                c.add_gate(gate, &[q]);
                t_left -= 1;
            }
        }
        for _ in 0..pairs {
            let a = rng.random_range(0..n);
            let b = (a + rng.random_range(1..n)) % n;
            c.add_gate(Gate::Cx, &[a, b]);
        }
    }
    c
}

/// The distribution of the classical bits under the circuit's measurement map, a later
/// measurement into a bit winning.
fn measured_distribution(circuit: &Circuit) -> Vec<f64> {
    let full = gate_only_statevector_probabilities(circuit);
    let mut writer = vec![None; circuit.num_classical_bits];
    for inst in &circuit.instructions {
        if let Instruction::Measure {
            qubit,
            classical_bit,
        } = inst
        {
            writer[*classical_bit] = Some(*qubit);
        }
    }
    let mut want = vec![0.0; 1 << circuit.num_classical_bits];
    for (state, p) in full.iter().enumerate() {
        let key = writer.iter().enumerate().fold(0usize, |acc, (cb, q)| {
            acc | q.map_or(0, |q| ((state >> q) & 1) << cb)
        });
        want[key] += p;
    }
    want
}

/// What a plan exercises, for the coverage the tier tests require.
#[derive(Default, Debug)]
struct Coverage {
    pack: usize,
    table: usize,
    cut_pairs: [usize; 4],
    mixed: usize,
    y_anchor: usize,
    dependent_row: usize,
    partial_rank: usize,
    tdg: usize,
    coins: usize,
}

impl Coverage {
    fn record(&mut self, circuit: &Circuit, plan: &SplitPlan) {
        let mut kinds = [false; 3];
        for part in &plan.parts {
            let width = part.mask.count_ones() as usize;
            match part.cut {
                Some(cut) => {
                    self.cut_pairs[cut.pairs.min(3)] += 1;
                    kinds[2] = true;
                }
                None if part.rank == 0 => {}
                None if part.rank <= plan.limits.pack_bits => {
                    self.pack += 1;
                    kinds[0] = true;
                }
                None => {
                    self.table += 1;
                    kinds[1] = true;
                }
            }
            if part.rank < width {
                self.partial_rank += 1;
            }
        }
        if kinds.iter().filter(|&&k| k).count() > 1 {
            self.mixed += 1;
        }
        if plan.gauge.anchors.iter().any(|[_, one]| one.im == 0.0) {
            self.y_anchor += 1;
        }
        let rank = row_basis(plan.register_rows.iter().map(|&(_, p)| p)).len();
        if plan.register_rows.len() > rank {
            self.dependent_row += 1;
        }
        if circuit.instructions.iter().any(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::Tdg,
                    ..
                }
            )
        }) {
            self.tdg += 1;
        }
        if plan.rows.uniform.iter().any(|&u| u) {
            self.coins += 1;
        }
    }
}

/// Random circuits of 10 to 14 qubits, every qubit or a subset measured, some into
/// overwritten bits.
fn tier_circuits() -> Vec<(String, Circuit)> {
    let mut circuits = Vec::new();
    for seed in 0..160u64 {
        let n = 10 + (seed as usize % 5);
        let t_count = 4 + (seed as usize / 5) % 7;
        let pairs = 1 + (seed as usize / 35) % 3;
        let mut circuit = layered_clifford_t(n, 4, t_count, pairs, seed);
        match seed % 4 {
            0 | 1 => circuit.measure_all(),
            2 => {
                for q in (0..n).step_by(2) {
                    circuit.add_measure(q, q / 2);
                }
            }
            _ => {
                for q in 0..n {
                    circuit.add_measure(q, q % (n - 2));
                }
            }
        }
        circuits.push((format!("seed {seed}, n {n}, t {t_count}"), circuit));
    }
    for seed in 0..160u64 {
        let n = 10 + (seed as usize % 3);
        let t_count = 5 + (seed as usize / 3) % 5;
        let pairs = 2 + (seed as usize / 15) % 4;
        let depth = 3 + (seed as usize / 60);
        let mut circuit = layered_clifford_t(n, depth, t_count, pairs, 1000 + seed);
        circuit.measure_all();
        circuits.push((format!("dense seed {seed}, n {n}, t {t_count}"), circuit));
    }
    for seed in 0..200u64 {
        let n = 10 + (seed as usize % 3);
        let t_count = 5 + (seed as usize / 3) % 5;
        let mut circuit = scrambled_clifford_t(n, 3 * n, t_count, 2000 + seed);
        circuit.measure_all();
        circuits.push((
            format!("scrambled seed {seed}, n {n}, t {t_count}"),
            circuit,
        ));
    }
    for seed in 0..400u64 {
        let (size, blocks, t_count) =
            [(4, 2, 6), (5, 2, 4), (2, 4, 3), (3, 3, 4)][seed as usize % 4];
        let n = size * blocks;
        let mut rng = ChaCha8Rng::seed_from_u64(5000 + seed);
        let mut circuit = Circuit::new(n, n);
        for block in 0..blocks {
            let inner =
                scrambled_clifford_t(size, 2 * size, t_count, 3000 + 7 * seed + block as u64);
            for inst in inner.instructions {
                if let Instruction::Gate { gate, targets } = inst {
                    let shifted: Vec<usize> = targets.iter().map(|q| q + block * size).collect();
                    circuit.add_gate(gate, &shifted);
                }
            }
        }
        for _ in 0..1 + (seed as usize / 4) % 6 {
            let a = rng.random_range(0..n);
            let b = (a / size * size + size + rng.random_range(0..size)) % n;
            circuit.add_gate(Gate::Cx, &[a, b]);
            let q = rng.random_range(0..n);
            circuit.add_gate([Gate::H, Gate::S][rng.random_range(0..2)].clone(), &[q]);
        }
        circuit.measure_all();
        circuits.push((
            format!("blocks seed {seed}, {blocks} x {size}q, t {t_count}"),
            circuit,
        ));
    }
    for seed in 0..400u64 {
        let units: &[usize] =
            [&[1, 1, 1, 3][..], &[4, 4], &[2, 2, 2, 2], &[1, 1, 2, 3]][seed as usize % 4];
        let circuits_gates = 6 + (seed as usize / 4) % 30;
        let circuit = linked_units(units, circuits_gates, 6000 + seed);
        circuits.push((
            format!("units seed {seed}, {units:?}, {circuits_gates} gates"),
            circuit,
        ));
    }
    circuits
}

/// One anchor per qubit, the qubits of each unit in `units` linked by a chain of excess
/// rotations, then `gates` random `H`, `S` or CX over every qubit, every qubit measured.
fn linked_units(units: &[usize], gates: usize, seed: u64) -> Circuit {
    let n: usize = units.iter().sum();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    let mut first = 0;
    for &width in units {
        for q in first..first + width - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
            c.add_gate(Gate::Tdg, &[q + 1]);
        }
        first += width;
    }
    for _ in 0..gates {
        let a = rng.random_range(0..n);
        match rng.random_range(0..3) {
            0 => c.add_gate(Gate::H, &[a]),
            1 => c.add_gate(Gate::S, &[a]),
            _ => c.add_gate(Gate::Cx, &[a, (a + rng.random_range(1..n)) % n]),
        }
    }
    c.measure_all();
    c
}

/// `H` on every qubit, then per T gate `gates` random `H`, `S`, `Sdg` or CX ahead of one T
/// or Tdg on a random qubit: the recipe of the scrambled bench rows.
fn scrambled_clifford_t(n: usize, gates: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for _ in 0..t_count {
        for _ in 0..gates {
            let choice = rng.random_range(0..4);
            if choice < 3 {
                let gate = [Gate::H, Gate::S, Gate::Sdg][choice].clone();
                c.add_gate(gate, &[rng.random_range(0..n)]);
            } else {
                let a = rng.random_range(0..n);
                let b = (a + rng.random_range(1..n)) % n;
                c.add_gate(Gate::Cx, &[a, b]);
            }
        }
        let gate = if rng.random_bool(0.5) {
            Gate::T
        } else {
            Gate::Tdg
        };
        c.add_gate(gate, &[rng.random_range(0..n)]);
    }
    c
}

#[test]
fn every_tier_draws_the_statevector_distribution() {
    let mut coverage = Coverage::default();
    let mut admitted = 0;
    for (label, circuit) in tier_circuits() {
        let Some(plan) = SplitPlan::with_limits(&circuit, TIER_LIMITS).unwrap() else {
            continue;
        };
        admitted += 1;
        coverage.record(&circuit, &plan);
        let got = plan.sampler().distribution();
        assert_close(&label, &got, &measured_distribution(&circuit), 1e-10);
    }
    assert!(admitted >= 600, "{admitted} admitted");
    // A cut needs every row independent on the part. A side whose forms sum to zero would
    // then hold rows of its own, splitting the rows in two, and the canonical merge would
    // already have made it a part: no cut has zero pairs.
    let c = &coverage;
    assert_eq!(c.cut_pairs[0], 0, "{coverage:?}");
    assert!(
        c.pack > 0
            && c.table > 0
            && c.cut_pairs[1] > 0
            && c.cut_pairs[2] > 0
            && c.cut_pairs[3] > 0
            && c.mixed > 0
            && c.y_anchor > 0
            && c.dependent_row > 0
            && c.partial_rank > 0
            && c.tdg > 0
            && c.coins > 0,
        "{coverage:?}"
    );
}

#[test]
fn default_limits_draw_the_statevector_distribution() {
    for (label, circuit) in tier_circuits().into_iter().step_by(3) {
        let sampler = SplitSampler::compile(&circuit)
            .unwrap()
            .expect("every part fits at the default limits");
        assert_close(
            &label,
            &sampler.distribution(),
            &measured_distribution(&circuit),
            1e-10,
        );
    }
}

/// Chi-square of `shots` against `want` over the outcomes with positive probability.
fn chi_square(shots: &[Vec<bool>], want: &[f64]) -> (f64, usize) {
    let mut seen = vec![0usize; want.len()];
    for shot in shots {
        let key = shot
            .iter()
            .enumerate()
            .fold(0usize, |acc, (i, &b)| acc | usize::from(b) << i);
        seen[key] += 1;
    }
    let n = shots.len() as f64;
    let mut stat = 0.0;
    let mut cells = 0;
    for (&count, &p) in seen.iter().zip(want) {
        if p > 0.0 {
            stat += (count as f64 - n * p).powi(2) / (n * p);
            cells += 1;
        } else {
            assert_eq!(count, 0, "a shot landed on an outcome of probability zero");
        }
    }
    (stat, cells.max(2) - 1)
}

#[test]
fn shots_follow_the_exact_distribution_on_every_tier() {
    let mut kinds = [false; 5];
    for (label, circuit) in tier_circuits() {
        let Some(plan) = SplitPlan::with_limits(&circuit, TIER_LIMITS).unwrap() else {
            continue;
        };
        let kind = plan
            .parts
            .iter()
            .map(|part| match part.cut {
                Some(cut) => 1 + cut.pairs.min(3),
                None => 0,
            })
            .max()
            .unwrap_or(0);
        if kinds[kind] {
            continue;
        }
        kinds[kind] = true;
        let want = measured_distribution(&circuit);
        let shots = plan.sampler().sample(100_000, 42);
        let (stat, df) = chi_square(&shots, &want);
        let bound = df as f64 + 6.0 * (2.0 * df as f64).sqrt();
        assert!(stat < bound, "{label}: chi-square {stat} on {df} degrees");
    }
    assert_eq!(
        kinds,
        [true, false, true, true, true],
        "whole, and cuts at 1 to 3 pairs"
    );
}

/// Counts the words a shot reads.
struct Counting {
    inner: ChaCha8Rng,
    words: usize,
}

impl rand::TryRng for Counting {
    type Error = std::convert::Infallible;

    fn try_next_u32(&mut self) -> std::result::Result<u32, Self::Error> {
        unreachable!("the sampler reads whole words")
    }

    fn try_next_u64(&mut self) -> std::result::Result<u64, Self::Error> {
        self.words += 1;
        Ok(self.inner.next_u64())
    }

    fn try_fill_bytes(&mut self, _: &mut [u8]) -> std::result::Result<(), Self::Error> {
        unreachable!("the sampler reads whole words")
    }
}

#[test]
fn every_shot_reads_the_same_words_from_its_block_stream() {
    for (label, circuit) in tier_circuits().into_iter().step_by(7) {
        let Some(plan) = SplitPlan::with_limits(&circuit, TIER_LIMITS).unwrap() else {
            continue;
        };
        let coins = plan.rows.uniform.iter().filter(|&&u| u).count();
        let sampler = plan.sampler();
        let tables = sampler
            .draws
            .iter()
            .filter(|d| matches!(d, Draw::Table { .. }))
            .count();
        let cuts = sampler.draws.len() - tables;
        let words = coins.div_ceil(64) + tables + 3 * cuts;
        assert_eq!(sampler.words_per_shot(), words, "{label}");

        let shots = sampler.sample(300, 7);
        let mut inner = ChaCha8Rng::seed_from_u64(7);
        inner.set_stream(3);
        let mut rng = Counting { inner, words: 0 };
        let mut scratch = sampler.scratch();
        let mut key = vec![0u64; sampler.base.len()];
        for (s, shot) in shots[256..].iter().enumerate() {
            sampler.draw(&mut rng, &mut scratch, &mut key);
            assert_eq!(rng.words, words * (s + 1), "{label}: shot {}", 256 + s);
            let bits: Vec<bool> = (0..sampler.num_classical_bits)
                .map(|cb| (key[cb >> 6] >> (cb & 63)) & 1 == 1)
                .collect();
            assert_eq!(&bits, shot, "{label}: shot {}", 256 + s);
        }
    }
}

#[test]
fn a_wide_register_from_one_rotated_qubit_draws_its_two_outcomes() {
    let n = 200;
    let mut circuit = Circuit::new(n, n);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    circuit.measure_all();
    let shots = SplitSampler::compile(&circuit)
        .unwrap()
        .unwrap()
        .sample(20_000, 42);
    assert!(shots.iter().all(|s| s.iter().all(|&b| b == s[0])));
    let zeros = shots.iter().filter(|s| !s[0]).count() as f64 / shots.len() as f64;
    let want = std::f64::consts::FRAC_PI_8.cos().powi(2);
    assert!(
        (zeros - want).abs() < 0.01,
        "P(0...0) = {zeros}, want {want}"
    );
}

#[test]
fn the_cut_search_is_a_function_of_its_forms() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for m in [5, 12, 13, 20] {
        let units: Vec<u64> = (0..m).map(|i| 1u64 << i).collect();
        let forms: Vec<Vec<u64>> = (0..m)
            .map(|_| {
                let mut form = vec![0u64; 10];
                for i in 0..10 {
                    for j in i + 1..10 {
                        if rng.random_bool(0.3) {
                            form[i] |= 1 << j;
                            form[j] |= 1 << i;
                        }
                    }
                }
                form
            })
            .collect();
        let refs: Vec<&[u64]> = forms.iter().map(|f| &f[..]).collect();
        let first = frozen_cut(&units, &refs).unwrap();
        assert_ne!(
            first.left & 1,
            0,
            "{m} units: the left side holds the first unit"
        );
        for _ in 0..100 {
            assert_eq!(frozen_cut(&units, &refs), Some(first), "{m} units");
        }
    }
}

/// `rotated` anchors, then one T on qubit 0 after CX gates from the others: an excess
/// rotation over every register qubit.
fn one_wide_unit(rotated: usize, extra: usize) -> Circuit {
    let n = rotated + extra;
    let mut circuit = Circuit::new(n, n);
    for q in 0..rotated {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 1..rotated {
        circuit.add_gate(Gate::Cx, &[q, 0]);
    }
    circuit.add_gate(Gate::T, &[0]);
    circuit.measure_all();
    circuit
}

#[test]
fn the_split_declines_past_its_anchors_its_widest_unit_and_its_pairs() {
    let anchored = |n: usize| {
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
            circuit.add_gate(Gate::T, &[q]);
        }
        circuit.measure_all();
        circuit
    };
    assert!(SplitPlan::new(&anchored(64)).unwrap().is_some());
    assert!(SplitPlan::new(&anchored(65)).unwrap().is_none());
    assert!(SplitPlan::new(&one_wide_unit(26, 1)).unwrap().is_some());
    assert!(SplitPlan::new(&one_wide_unit(27, 1)).unwrap().is_none());

    // A cut past its pairs, at the tier limits with one pair allowed, and a part past 6
    // qubits read on fewer rows than its width, each admitted once that one limit is
    // raised.
    let (mut pairs, mut partial) = (0, 0);
    let one_pair = SplitLimits {
        max_pairs: 1,
        ..TIER_LIMITS
    };
    for seed in 0..200u64 {
        let circuit = linked_units(&[4, 4], 30, 7000 + seed);
        if SplitPlan::with_limits(&circuit, one_pair)
            .unwrap()
            .is_none()
            && let Some(plan) = SplitPlan::with_limits(&circuit, TIER_LIMITS).unwrap()
        {
            assert!(
                plan.parts
                    .iter()
                    .any(|p| p.cut.is_some_and(|c| c.pairs > 1))
            );
            pairs += 1;
        }
    }
    let wider = SplitLimits {
        table_width: 7,
        total_log2: 8,
        ..TIER_LIMITS
    };
    for seed in 0..200u64 {
        let mut circuit = linked_units(&[4, 3], 30, 8000 + seed);
        let first = circuit
            .instructions
            .iter()
            .position(|i| matches!(i, Instruction::Measure { .. }))
            .unwrap();
        circuit.instructions.remove(first);
        if SplitPlan::with_limits(&circuit, TIER_LIMITS)
            .unwrap()
            .is_none()
            && let Some(plan) = SplitPlan::with_limits(&circuit, wider).unwrap()
            && plan
                .parts
                .iter()
                .any(|p| p.cut.is_none() && p.mask.count_ones() == 7 && p.rank < 7)
        {
            partial += 1;
        }
    }
    assert!(
        pairs > 0 && partial > 0,
        "{pairs} past the pairs, {partial} partial"
    );
}

#[test]
fn walsh_hadamard_matches_the_radix_2_sweep_bit_for_bit() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for r in [0, 1, 2, 3, 4, 5, 10, 11, 12, 14, 15, 16, 17, 18, 20] {
        let values: Vec<f64> = (0..1usize << r)
            .map(|_| rng.random::<f64>() - 0.5)
            .collect();
        let mut want = values.clone();
        walsh(&mut want);
        let mut got = values;
        walsh_hadamard(&mut got);
        assert_eq!(got, want, "2^{r} entries");
    }
}

/// Every whole part with a row of each circuit, as the plan and the part's basis.
fn whole_parts(plan: &SplitPlan) -> Vec<(&Part, Vec<Letters>)> {
    plan.parts
        .iter()
        .filter(|part| part.cut.is_none() && part.rank > 0)
        .map(|part| {
            let mut pieces: Vec<Letters> = plan
                .register_rows
                .iter()
                .map(|(_, p)| (p.x & part.mask, p.z & part.mask))
                .collect();
            (part, ordered_basis(&mut pieces, part.mask))
        })
        .collect()
}

#[test]
fn walsh_tables_match_the_dense_tables() {
    let (mut excess, mut partial) = (0, 0);
    // One measurement dropped from two linked units leaves a part read on fewer rows than
    // its width.
    let partial_rank = (0..40u64).map(|seed| {
        let mut circuit = linked_units(&[4, 3], 30, 8000 + seed);
        let first = circuit
            .instructions
            .iter()
            .position(|i| matches!(i, Instruction::Measure { .. }))
            .unwrap();
        circuit.instructions.remove(first);
        (format!("partial seed {seed}"), circuit, SPLIT_LIMITS)
    });
    let tiers = tier_circuits()
        .into_iter()
        .map(|(label, circuit)| (label, circuit, TIER_LIMITS));
    for (label, circuit, limits) in tiers.chain(partial_rank) {
        let Some(plan) = SplitPlan::with_limits(&circuit, limits).unwrap() else {
            continue;
        };
        for (part, basis) in whole_parts(&plan) {
            excess += usize::from(!part.excess.is_empty());
            partial += usize::from(part.rank < part.mask.count_ones() as usize);
            let walsh = plan.walsh_table(part, &basis);
            let dense = plan.dense_table(part, &basis);
            assert_eq!(walsh.len(), dense.len(), "{label}");
            for (t, (w, d)) in walsh.iter().zip(&dense).enumerate() {
                assert!((w - d).abs() < 1e-12, "{label}: entry {t}: {w} against {d}");
            }
        }
    }
    assert!(
        excess > 0 && partial > 0,
        "{excess} with excess, {partial} partial"
    );
}

/// `plan` with every whole part read out from its dense amplitudes, as before the Walsh
/// readout.
fn dense_route(mut plan: SplitPlan) -> SplitPlan {
    for part in &mut plan.parts {
        part.walsh = false;
    }
    plan
}

#[test]
fn walsh_readout_draws_the_dense_route_distribution() {
    let mut walsh_parts = 0;
    for (label, circuit) in tier_circuits().into_iter().step_by(5) {
        let Some(plan) = SplitPlan::new(&circuit).unwrap() else {
            continue;
        };
        walsh_parts += plan.parts.iter().filter(|p| p.walsh).count();
        let want = dense_route(SplitPlan::new(&circuit).unwrap().unwrap())
            .sampler()
            .distribution();
        let got = plan.sampler().distribution();
        let distance: f64 = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).sum();
        assert!(distance < 1e-11, "{label}: total variation {distance}");
    }
    assert!(walsh_parts > 0);

    for seed in 0..4u64 {
        let mut circuit = scrambled_clifford_t(12, 36, 12, 9000 + seed);
        circuit.measure_all();
        let plan = SplitPlan::new(&circuit).unwrap().unwrap();
        assert!(
            plan.parts.iter().any(|p| p.walsh && p.rank >= 8),
            "seed {seed}"
        );
        let want = dense_route(SplitPlan::new(&circuit).unwrap().unwrap())
            .sampler()
            .distribution();
        let shots = plan.sampler().sample(100_000, 42);
        let (stat, df) = chi_square(&shots, &want);
        let bound = df as f64 + 6.0 * (2.0 * df as f64).sqrt();
        assert!(
            stat < bound,
            "seed {seed}: chi-square {stat} on {df} degrees"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn walsh_readout_is_identical_on_one_and_many_threads() {
    let circuit = (0..20u64)
        .map(|seed| {
            let mut circuit = scrambled_clifford_t(18, 54, 18, 9100 + seed);
            circuit.measure_all();
            circuit
        })
        .find(|circuit| {
            SplitPlan::new(circuit)
                .unwrap()
                .is_some_and(|plan| plan.parts.iter().any(|p| p.walsh && p.rank >= 16))
        })
        .expect("a whole part of at least 16 rows");
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let plan = SplitPlan::new(&circuit).unwrap().unwrap();
                let tables: Vec<Vec<f64>> = whole_parts(&plan)
                    .into_iter()
                    .map(|(part, basis)| plan.walsh_table(part, &basis))
                    .collect();
                (tables, plan.sampler().sample(4096, 42))
            })
    };
    let (one, many) = (run(1), run(4));
    assert_eq!(one.0, many.0);
    assert_eq!(one.1, many.1);
}

/// Two units of `a` and `b` anchored qubits, each linked by one excess rotation over every
/// qubit of the unit, then `gates` random CX across the units, `S` or `H`, every qubit
/// measured in the `X` basis.
fn two_linked_units(a: usize, b: usize, gates: usize, seed: u64) -> Circuit {
    let n = a + b;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    for (first, width) in [(0, a), (a, b)] {
        for q in first + 1..first + width {
            c.add_gate(Gate::Cx, &[q, first]);
        }
        c.add_gate(Gate::T, &[first]);
        for q in first + 1..first + width {
            c.add_gate(Gate::Cx, &[q, first]);
        }
    }
    for _ in 0..gates {
        let x = rng.random_range(0..a);
        let y = a + rng.random_range(0..b);
        match rng.random_range(0..4) {
            0 => c.add_gate(Gate::Cx, &[x, y]),
            1 => c.add_gate(Gate::Cx, &[y, x]),
            2 => c.add_gate(Gate::S, &[rng.random_range(0..n)]),
            _ => c.add_gate(Gate::H, &[rng.random_range(0..n)]),
        }
    }
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    c.measure_all();
    c
}

/// [`SPLIT_LIMITS`] with every part up to the table width held whole ahead of a cut.
const WHOLE_LIMITS: SplitLimits = SplitLimits {
    cheap_pairs: 0,
    whole_first_width: SPLIT_LIMITS.table_width,
    ..SPLIT_LIMITS
};

/// The plan of `circuit` at `limits` with its one part of `width` qubits, asserting that
/// the part is cut or held whole as `cut` says.
fn plan_with_wide_part(circuit: &Circuit, limits: SplitLimits, width: u32, cut: bool) -> SplitPlan {
    let plan = SplitPlan::with_limits(circuit, limits).unwrap().unwrap();
    let wide: Vec<&Part> = plan
        .parts
        .iter()
        .filter(|p| p.mask.count_ones() == width)
        .collect();
    assert_eq!(wide.len(), 1);
    assert_eq!(wide[0].cut.is_some(), cut);
    assert!(cut || wide[0].walsh);
    plan
}

#[test]
fn a_part_past_24_qubits_takes_the_cut_that_prices_lower() {
    // Nine pairs across the cut, past the cheap pairs: up to 24 qubits the part would be
    // held whole, and here the cut prices far below the 2^26 entry readout.
    let circuit = two_linked_units(13, 13, 128, 42);
    let plan = plan_with_wide_part(&circuit, SPLIT_LIMITS, 26, true);
    let part = plan.parts.iter().find(|p| p.cut.is_some()).unwrap();
    assert!(part.cut.unwrap().pairs > SPLIT_LIMITS.cheap_pairs);
    plan_with_wide_part(&circuit, WHOLE_LIMITS, 26, false);
}

/// Two-sample chi-square of equally many shots `a` and `b` on the classical bits `bits`,
/// over the cells either reaches.
fn two_sample_chi_square(
    a: &[Vec<bool>],
    b: &[Vec<bool>],
    bits: std::ops::Range<usize>,
) -> (f64, usize) {
    assert_eq!(a.len(), b.len());
    let cells = 1usize << bits.len();
    let count = |shots: &[Vec<bool>]| {
        let mut seen = vec![0usize; cells];
        for shot in shots {
            let key = shot[bits.clone()]
                .iter()
                .enumerate()
                .fold(0usize, |acc, (i, &bit)| acc | usize::from(bit) << i);
            seen[key] += 1;
        }
        seen
    };
    let (x, y) = (count(a), count(b));
    let mut stat = 0.0;
    let mut reached = 0;
    for (&x, &y) in x.iter().zip(&y) {
        if x + y > 0 {
            stat += (x as f64 - y as f64).powi(2) / (x + y) as f64;
            reached += 1;
        }
    }
    (stat, reached.max(2) - 1)
}

#[test]
fn a_whole_part_of_26_qubits_draws_the_cut_distribution() {
    let circuit = two_linked_units(13, 13, 64, 42);
    let cut = plan_with_wide_part(&circuit, SPLIT_LIMITS, 26, true);
    let whole = plan_with_wide_part(&circuit, WHOLE_LIMITS, 26, false);
    let shots = 1 << 18;
    let from_whole = whole.sampler().sample(shots, 42);
    let from_cut = cut.sampler().sample(shots, 7);
    // Twelve bits across both units, far from uniform on this circuit.
    let (stat, df) = two_sample_chi_square(&from_whole, &from_cut, 7..19);
    assert!(df > 1000, "{df} cells reached");
    let bound = df as f64 + 6.0 * (2.0 * df as f64).sqrt();
    assert!(stat < bound, "chi-square {stat} on {df} degrees");
}

#[cfg(feature = "parallel")]
#[test]
fn a_whole_part_of_26_qubits_is_identical_on_one_and_many_threads() {
    let circuit = two_linked_units(13, 13, 64, 42);
    let pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    };
    let plan = plan_with_wide_part(&circuit, WHOLE_LIMITS, 26, false);
    let (part, basis) = whole_parts(&plan)
        .into_iter()
        .find(|(part, _)| part.mask.count_ones() == 26)
        .unwrap();
    let one = pool(1).install(|| plan.walsh_table(part, &basis));
    assert!(one == pool(4).install(|| plan.walsh_table(part, &basis)));
    drop(one);
    let shots = |threads: usize| {
        pool(threads).install(|| {
            plan_with_wide_part(&circuit, WHOLE_LIMITS, 26, false)
                .sampler()
                .sample(4096, 42)
        })
    };
    assert_eq!(shots(1), shots(4));
}

fn unpruned_value(
    readout: &WalshReadout,
    q: RegisterPauli,
    rotations: usize,
    calls: &mut usize,
) -> f64 {
    *calls += 1;
    if rotations == 0 {
        return readout.product_value(q);
    }
    let (r, cos, sin) = readout.excess[rotations - 1];
    if ((q.x & r.z) ^ (q.z & r.x)).count_ones() & 1 == 0 {
        return unpruned_value(readout, q, rotations - 1, calls);
    }
    let mut qr = q.mul(r);
    qr.phase4 = (qr.phase4 + 3) & 3;
    cos * unpruned_value(readout, q, rotations - 1, calls)
        + sin * unpruned_value(readout, qr, rotations - 1, calls)
}

fn counted_pruned_value(
    readout: &WalshReadout,
    q: RegisterPauli,
    rotations: usize,
    calls: &mut usize,
) -> f64 {
    *calls += 1;
    if rotations == 0 {
        return readout.product_value(q);
    }
    let xs = q.x & !q.z;
    let ys = q.x & q.z;
    if (xs & readout.frozen_blind_x[rotations]) | (ys & readout.frozen_blind_y[rotations]) != 0 {
        return 0.0;
    }
    let (r, cos, sin) = readout.excess[rotations - 1];
    if ((q.x & r.z) ^ (q.z & r.x)).count_ones() & 1 == 0 {
        return counted_pruned_value(readout, q, rotations - 1, calls);
    }
    let mut qr = q.mul(r);
    qr.phase4 = (qr.phase4 + 3) & 3;
    cos * counted_pruned_value(readout, q, rotations - 1, calls)
        + sin * counted_pruned_value(readout, qr, rotations - 1, calls)
}

fn unpruned_walsh_table(readout: &WalshReadout, rank: usize) -> Vec<f64> {
    let mut table = vec![0.0f64; 1 << rank];
    let mut calls = 0usize;
    for (s, entry) in table.iter_mut().enumerate() {
        let q = readout.product(s);
        *entry = unpruned_value(readout, q, readout.excess.len(), &mut calls);
    }
    walsh_hadamard(&mut table);
    let scale = 0.5f64.powi(rank as i32);
    for p in &mut table {
        *p = (*p * scale).max(0.0);
    }
    table
}

fn assert_all_paulis_bit_exact(readout: &WalshReadout, width: usize, label: &str) {
    let limit = 1usize << width;
    let mut unpruned_calls = 0usize;
    let mut pruned_calls = 0usize;
    for k in 0..=readout.excess.len() {
        for x in 0..limit {
            for z in 0..limit {
                for phase4 in [0u8, 2u8] {
                    let q = RegisterPauli {
                        x: x as u64,
                        z: z as u64,
                        phase4,
                    };
                    let got = readout.value(q, k);
                    let want = unpruned_value(readout, q, k, &mut unpruned_calls);
                    let counted = counted_pruned_value(readout, q, k, &mut pruned_calls);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{label}: k={k}, x={x:#x}, z={z:#x}, phase4={phase4}: {got} vs {want}"
                    );
                    assert_eq!(counted.to_bits(), got.to_bits(), "{label}");
                }
            }
        }
    }
}

#[test]
fn walsh_pruning_matches_unpruned_on_tiers_and_partial_rank() {
    let (mut excess, mut partial) = (0, 0);
    let partial_rank = (0..40u64).map(|seed| {
        let mut circuit = linked_units(&[4, 3], 30, 8000 + seed);
        let first = circuit
            .instructions
            .iter()
            .position(|i| matches!(i, Instruction::Measure { .. }))
            .unwrap();
        circuit.instructions.remove(first);
        (format!("partial seed {seed}"), circuit, SPLIT_LIMITS)
    });
    let tiers = tier_circuits()
        .into_iter()
        .map(|(label, circuit)| (label, circuit, TIER_LIMITS));
    for (label, circuit, limits) in tiers.chain(partial_rank) {
        let Some(plan) = SplitPlan::with_limits(&circuit, limits).unwrap() else {
            continue;
        };
        for (part, basis) in whole_parts(&plan) {
            excess += usize::from(!part.excess.is_empty());
            partial += usize::from(part.rank < part.mask.count_ones() as usize);
            let readout = WalshReadout::new(&plan.gauge, part.mask, &part.excess, &basis);
            assert_eq!(readout.frozen_blind_x.len(), readout.excess.len() + 1);
            assert_eq!(readout.frozen_blind_y.len(), readout.excess.len() + 1);
            let mut calls = 0usize;
            for s in 0..1usize << basis.len() {
                let q = readout.product(s);
                let got = readout.value(q, readout.excess.len());
                let want = unpruned_value(&readout, q, readout.excess.len(), &mut calls);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{label}: subset {s}: {got} vs {want}"
                );
            }
            let got_table = plan.walsh_table(part, &basis);
            let want_table = unpruned_walsh_table(&readout, basis.len());
            assert_eq!(got_table.len(), want_table.len(), "{label}");
            for (t, (g, w)) in got_table.iter().zip(&want_table).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{label}: table entry {t}: {g} vs {w}"
                );
            }
            let dense = plan.dense_table(part, &basis);
            for (t, (g, d)) in got_table.iter().zip(&dense).enumerate() {
                assert!((g - d).abs() < 1e-12, "{label}: entry {t}: {g} vs {d}");
            }
        }
    }
    assert!(
        excess > 0 && partial > 0,
        "{excess} with excess, {partial} partial"
    );
}

#[test]
fn walsh_pruning_handles_zero_excess_and_all_x_or_all_y_anchors() {
    let build_four = |y_mask: u8, with_excess: bool| {
        let mut c = Circuit::new(4, 4);
        for q in 0..4 {
            if (y_mask >> q) & 1 == 1 {
                c.add_gate(Gate::S, &[q]);
            }
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::T, &[q]);
        }
        if with_excess {
            c.add_gate(Gate::Cx, &[0, 1]);
            c.add_gate(Gate::T, &[1]);
            c.add_gate(Gate::Cx, &[0, 1]);
            c.add_gate(Gate::Cx, &[1, 2]);
            c.add_gate(Gate::Tdg, &[2]);
            c.add_gate(Gate::Cx, &[1, 2]);
            c.add_gate(Gate::Cx, &[2, 3]);
            c.add_gate(Gate::T, &[3]);
            c.add_gate(Gate::Cx, &[2, 3]);
        }
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[0, 2]);
        c.add_gate(Gate::Cx, &[0, 3]);
        c.add_gate(Gate::H, &[0]);
        c.measure_all();
        c
    };

    // Zero excess rotations (rotations == 0) with mixed X and Y blind anchors.
    let zero_excess = build_four(0b1100, false);
    let plan = SplitPlan::new(&zero_excess).unwrap().unwrap();
    let parts = whole_parts(&plan);
    assert_eq!(parts.len(), 1);
    let (part, basis) = &parts[0];
    assert!(part.excess.is_empty());
    assert_eq!(part.mask, 0b1111);
    let readout = WalshReadout::new(&plan.gauge, part.mask, &part.excess, basis);
    assert_eq!(readout.frozen_blind_x, vec![readout.blind[0]]);
    assert_eq!(readout.frozen_blind_y, vec![readout.blind[1]]);
    assert_all_paulis_bit_exact(&readout, 4, "zero_excess");
    let got_table = plan.walsh_table(part, basis);
    let want_table = unpruned_walsh_table(&readout, basis.len());
    for (g, w) in got_table.iter().zip(&want_table) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    let dense = plan.dense_table(part, basis);
    assert_close("zero_excess", &got_table, &dense, 1e-12);

    // All-X (H; T) vs all-Y (S; H; T) blind anchors with excess rotations.
    for (label, y_mask, want_blind) in [
        ("all_x", 0b0000u8, [0b1111u64, 0u64]),
        ("all_y", 0b1111u8, [0u64, 0b1111u64]),
    ] {
        let circuit = build_four(y_mask, true);
        let plan = SplitPlan::new(&circuit).unwrap().unwrap();
        let parts = whole_parts(&plan);
        assert_eq!(parts.len(), 1, "{label}");
        let (part, basis) = &parts[0];
        assert_eq!(part.excess.len(), 3, "{label}");
        let readout = WalshReadout::new(&plan.gauge, part.mask, &part.excess, basis);
        assert_eq!(readout.blind, want_blind, "{label}");
        assert_all_paulis_bit_exact(&readout, 4, label);
        let got_table = plan.walsh_table(part, basis);
        let want_table = unpruned_walsh_table(&readout, basis.len());
        for (g, w) in got_table.iter().zip(&want_table) {
            assert_eq!(g.to_bits(), w.to_bits(), "{label}");
        }
        let dense = plan.dense_table(part, basis);
        assert_close(label, &got_table, &dense, 1e-12);
    }
}

#[test]
fn walsh_pruning_stages_frozen_blind_across_excess_rotations() {
    // Untouched anchor (qubit 3) merged into the same part via canonical_parts.
    let mut untouched = Circuit::new(4, 4);
    for q in 0..4 {
        if q == 1 {
            untouched.add_gate(Gate::S, &[q]);
        }
        untouched.add_gate(Gate::H, &[q]);
        untouched.add_gate(Gate::T, &[q]);
    }
    untouched.add_gate(Gate::Cx, &[0, 1]);
    untouched.add_gate(Gate::T, &[1]);
    untouched.add_gate(Gate::Cx, &[0, 1]);
    untouched.add_gate(Gate::Cx, &[0, 2]);
    untouched.add_gate(Gate::Tdg, &[2]);
    untouched.add_gate(Gate::Cx, &[0, 2]);
    untouched.add_gate(Gate::Cx, &[0, 1]);
    untouched.add_gate(Gate::Cx, &[0, 2]);
    untouched.add_gate(Gate::Cx, &[0, 3]);
    untouched.add_gate(Gate::H, &[0]);
    untouched.measure_all();

    let plan = SplitPlan::new(&untouched).unwrap().unwrap();
    let parts = whole_parts(&plan);
    assert_eq!(parts.len(), 1);
    let (part, basis) = &parts[0];
    assert_eq!(part.mask, 0b1111);
    assert_eq!(part.excess.len(), 2);
    let readout = WalshReadout::new(&plan.gauge, part.mask, &part.excess, basis);
    assert_eq!(readout.blind, [0b1101, 0b0010]);
    assert_eq!(readout.frozen_blind_x, vec![0b1101, 0b1100, 0b1000]);
    assert_eq!(readout.frozen_blind_y, vec![0b0010, 0b0000, 0b0000]);
    assert_all_paulis_bit_exact(&readout, 4, "untouched_merged");
    let got_table = plan.walsh_table(part, basis);
    let want_table = unpruned_walsh_table(&readout, basis.len());
    for (g, w) in got_table.iter().zip(&want_table) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    assert_close(
        "untouched_merged",
        &got_table,
        &plan.dense_table(part, basis),
        1e-12,
    );

    // Anchor 3 touched only by the last rotation excess[E - 1].
    let mut staged = Circuit::new(4, 4);
    for q in 0..4 {
        if q == 1 {
            staged.add_gate(Gate::S, &[q]);
        }
        staged.add_gate(Gate::H, &[q]);
        staged.add_gate(Gate::T, &[q]);
    }
    staged.add_gate(Gate::Cx, &[0, 1]);
    staged.add_gate(Gate::T, &[1]);
    staged.add_gate(Gate::Cx, &[0, 1]);
    staged.add_gate(Gate::Cx, &[1, 2]);
    staged.add_gate(Gate::Tdg, &[2]);
    staged.add_gate(Gate::Cx, &[1, 2]);
    staged.add_gate(Gate::Cx, &[2, 3]);
    staged.add_gate(Gate::T, &[3]);
    staged.add_gate(Gate::Cx, &[2, 3]);
    for q in 0..4 {
        staged.add_gate(Gate::H, &[q]);
    }
    staged.measure_all();

    let plan = SplitPlan::new(&staged).unwrap().unwrap();
    let parts = whole_parts(&plan);
    assert_eq!(parts.len(), 1);
    let (part, basis) = &parts[0];
    let readout = WalshReadout::new(&plan.gauge, part.mask, &part.excess, basis);
    assert_eq!(readout.frozen_blind_x, vec![0b1101, 0b1100, 0b1000, 0b0000]);
    assert_eq!(readout.frozen_blind_y, vec![0b0010, 0b0000, 0b0000, 0b0000]);
    assert_all_paulis_bit_exact(&readout, 4, "staged_last_rotation");
    let got_table = plan.walsh_table(part, basis);
    let want_table = unpruned_walsh_table(&readout, basis.len());
    for (g, w) in got_table.iter().zip(&want_table) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    assert_close(
        "staged_last_rotation",
        &got_table,
        &plan.dense_table(part, basis),
        1e-12,
    );
}

#[test]
fn walsh_pruning_works_above_64_circuit_qubits_via_high_bit_compress() {
    let n = 72;
    let mut circuit = Circuit::new(n, n);
    for q in 0..56 {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 56..66 {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 66..72 {
        if q == 67 {
            circuit.add_gate(Gate::S, &[q]);
        }
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit.add_gate(Gate::Cx, &[66, 67]);
    circuit.add_gate(Gate::T, &[67]);
    circuit.add_gate(Gate::Cx, &[66, 67]);
    circuit.add_gate(Gate::Cx, &[66, 68]);
    circuit.add_gate(Gate::Tdg, &[68]);
    circuit.add_gate(Gate::Cx, &[66, 68]);
    for q in 0..3 {
        circuit.add_gate(Gate::Cx, &[66 + q, 69 + q]);
        circuit.add_gate(Gate::H, &[66 + q]);
    }
    circuit.measure_all();

    let plan = SplitPlan::new(&circuit).unwrap().unwrap();
    let (part, basis) = whole_parts(&plan)
        .into_iter()
        .find(|(part, _)| part.mask >> 56 != 0)
        .expect("high-bit register part");
    assert_eq!(part.mask.count_ones(), 6);
    assert!(part.walsh);
    let readout = WalshReadout::new(&plan.gauge, part.mask, &part.excess, &basis);
    assert_eq!(readout.frozen_blind_x, vec![0b111101, 0b111100, 0b111000]);
    assert_eq!(readout.frozen_blind_y, vec![0b000010, 0b000000, 0b000000]);
    let mut calls = 0usize;
    for s in 0..1usize << basis.len() {
        let q = readout.product(s);
        for k in 0..=readout.excess.len() {
            let got = readout.value(q, k);
            let want = unpruned_value(&readout, q, k, &mut calls);
            assert_eq!(got.to_bits(), want.to_bits(), "s={s}, k={k}");
        }
    }
    let got_table = plan.walsh_table(part, &basis);
    let want_table = unpruned_walsh_table(&readout, basis.len());
    for (g, w) in got_table.iter().zip(&want_table) {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    assert_close(
        "above_64q_high_bit",
        &got_table,
        &plan.dense_table(part, &basis),
        1e-12,
    );
}

#[test]
fn walsh_pruning_reduces_visited_nodes_and_triggers_frozen_blind() {
    // Directly exercise WalshReadout::value's frozen_blind branch independently of blind[].
    let synthetic = WalshReadout {
        rows: Vec::new(),
        blind: [0, 0],
        frozen_blind_x: vec![0, 0b01],
        frozen_blind_y: vec![0, 0b10],
        negative: [0, 0, 0],
        scale: [1.0; 65],
        excess: vec![(
            RegisterPauli {
                x: 0,
                z: 0b01,
                phase4: 0,
            },
            FRAC_1_SQRT_2,
            FRAC_1_SQRT_2,
        )],
    };
    let x0 = RegisterPauli {
        x: 0b01,
        z: 0,
        phase4: 0,
    };
    let y1 = RegisterPauli {
        x: 0b10,
        z: 0b10,
        phase4: 0,
    };
    let z0 = RegisterPauli {
        x: 0,
        z: 0b01,
        phase4: 0,
    };
    assert_eq!(synthetic.value(x0, 1).to_bits(), 0.0f64.to_bits());
    assert_eq!(synthetic.value(y1, 1).to_bits(), 0.0f64.to_bits());
    assert_ne!(synthetic.value(z0, 1), 0.0);

    // Part with 6 anchors, 4 excess rotations on qubits 0..3, and 3 frozen anchors merged via canonical_parts.
    let mut circuit = Circuit::new(6, 6);
    for q in 0..6 {
        if q % 2 == 1 {
            circuit.add_gate(Gate::S, &[q]);
        }
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for &(a, b) in &[(0, 1), (1, 2), (0, 2), (0, 1)] {
        circuit.add_gate(Gate::Cx, &[a, b]);
        circuit.add_gate(Gate::T, &[b]);
        circuit.add_gate(Gate::Cx, &[a, b]);
    }
    for q in 0..3 {
        circuit.add_gate(Gate::Cx, &[q, q + 3]);
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.measure_all();

    let plan = SplitPlan::new(&circuit).unwrap().unwrap();
    let parts = whole_parts(&plan);
    assert_eq!(parts.len(), 1);
    let (part, basis) = &parts[0];
    assert_eq!(part.mask.count_ones(), 6);
    assert_eq!(part.excess.len(), 4);
    let readout = WalshReadout::new(&plan.gauge, part.mask, &part.excess, basis);
    let mut unpruned_calls = 0usize;
    let mut pruned_calls = 0usize;
    for s in 0..1usize << basis.len() {
        let q = readout.product(s);
        let got = readout.value(q, readout.excess.len());
        let want = unpruned_value(&readout, q, readout.excess.len(), &mut unpruned_calls);
        let counted = counted_pruned_value(&readout, q, readout.excess.len(), &mut pruned_calls);
        assert_eq!(got.to_bits(), want.to_bits());
        assert_eq!(counted.to_bits(), got.to_bits());
    }
    assert!(
        pruned_calls * 2 < unpruned_calls,
        "expected >2x call reduction from blind-bit pruning, got pruned={pruned_calls}, unpruned={unpruned_calls}"
    );
}

#[test]
fn walsh_readout_work_accounts_for_pruning_and_preserves_baselines() {
    // 0-excess preserves WALSH_TERM_UNITS.
    for (width, rank) in [(4usize, 4usize), (16, 12), (26, 26)] {
        let expected =
            2f64.powi(rank as i32) * (WALSH_TERM_UNITS + WALSH_LEVEL_UNITS * rank as f64);
        assert_eq!(walsh_readout_work(width, rank, []), expected);
    }

    // u_k == 0 for all k preserves 1.5^E.
    let width = 8usize;
    let rank = 8usize;
    let full = (1u64 << width) - 1;
    for excess in 1..=8usize {
        let got = walsh_readout_work(width, rank, std::iter::repeat_n(full, excess));
        let want = 2f64.powi(rank as i32)
            * (WALSH_TERM_UNITS * 1.5f64.powi(excess as i32) + WALSH_LEVEL_UNITS * rank as f64);
        assert!(
            (got - want).abs() < 1e-9,
            "excess={excess}: {got} vs {want}"
        );
    }

    // u_k > 0 scales expected surviving calls by (3/4)^{u_k}.
    let zero = walsh_readout_work(12, 12, []);
    let partial = walsh_readout_work(12, 12, std::iter::repeat_n(0b111u64, 6));
    let unpruned = walsh_readout_work(12, 12, std::iter::repeat_n((1u64 << 12) - 1, 6));
    assert!(zero < partial && partial < unpruned);
    let sum: f64 = (1..=6).map(|k| 1.5f64.powi(6 - k) * 0.75f64.powi(9)).sum();
    let expected_partial =
        2f64.powi(12) * (WALSH_TERM_UNITS * (1.0 + 0.5 * sum) + WALSH_LEVEL_UNITS * 12.0);
    assert!((partial - expected_partial).abs() < 1e-6);

    // Transform work WALSH_LEVEL_UNITS * rank stays intact so width 25..=26 cut choices hold.
    let min_wide = walsh_readout_work(26, 26, [0u64; 2]);
    assert!(min_wide > 1.1e9, "wide transform floor: {min_wide}");
    let plan = plan_with_wide_part(&two_linked_units(13, 13, 128, 42), SPLIT_LIMITS, 26, true);
    assert!(plan.parts.iter().any(|p| p.cut.is_some()));
}
