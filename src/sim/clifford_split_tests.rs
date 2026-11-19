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

use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Instruction;
use crate::gates::Gate;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The tier limits of the test plan, low enough that circuits of 10 to 14 qubits reach
/// every tier.
const TIER_LIMITS: SplitLimits = SplitLimits {
    pack_bits: 2,
    whole_width: 4,
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

/// The distribution of the classical bits under the circuit's measurement map, a later
/// measurement into a bit winning.
fn measured_distribution(circuit: &Circuit) -> Vec<f64> {
    let full = statevector_probabilities(circuit);
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

fn assert_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-10,
            "{label}: outcome {i}: got {g}, want {w}"
        );
    }
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
        assert_close(&got, &measured_distribution(&circuit), &label);
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
            &sampler.distribution(),
            &measured_distribution(&circuit),
            &label,
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
    assert!(SplitPlan::new(&one_wide_unit(24, 1)).unwrap().is_some());
    assert!(SplitPlan::new(&one_wide_unit(25, 1)).unwrap().is_none());

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
