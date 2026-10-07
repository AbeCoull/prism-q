//! Cross-route sampling contract: every shot and count route the crate dispatches
//! to must agree on the classical register a circuit writes, under permuted
//! measurement maps, unused bits, repeated writes, and registers past one word.

mod common;

use std::collections::HashMap;

use common::{SEED, sv_reference_probs};
use prism_q::circuit::SmallVec;
use prism_q::{
    BackendKind, Circuit, ClassicalCondition, Engine, Gate, Instruction, NoiseModel, PackedShots,
    ResolvedBackend, RunMetadata, compile_detector_sampler, compile_measurements,
    run_shots_compiled, simulate,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SHOTS: usize = 4000;

fn check_single_shot_independence(rank: usize) {
    let mut circuit = Circuit::new(rank + 1, rank + 1);
    for q in 0..rank {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::X, &[rank]);
    for q in 0..=rank {
        circuit.add_measure(q, q);
    }
    let mut sampler = compile_measurements(&circuit, SEED).unwrap();
    assert_eq!(sampler.rank(), rank);
    let mut ones = vec![0; rank];
    let mut joint = vec![[0usize; 4]; rank.saturating_sub(1)];
    for _ in 0..SHOTS {
        let sample = sampler.sample();
        assert_eq!(sample.len(), rank + 1);
        assert!(sample[rank]);
        for (q, count) in ones.iter_mut().enumerate() {
            *count += usize::from(sample[q]);
        }
        for (q, counts) in joint.iter_mut().enumerate() {
            counts[usize::from(sample[q]) + 2 * usize::from(sample[rank - 1])] += 1;
        }
    }
    for (q, count) in ones.into_iter().enumerate() {
        assert!(
            count.abs_diff(SHOTS / 2) < SHOTS / 16,
            "rank {rank}, qubit {q}: {count}"
        );
    }
    for (q, counts) in joint.into_iter().enumerate() {
        for count in counts {
            assert!(
                count.abs_diff(SHOTS / 4) < SHOTS / 16,
                "rank {rank}, qubits {q} and {}: {counts:?}",
                rank - 1,
            );
        }
    }
}

macro_rules! single_shot_cases {
    ($($name:ident => $rank:literal),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                check_single_shot_independence($rank);
            }
        )+
    };
}

single_shot_cases! {
    single_shot_rank_0 => 0,
    single_shot_rank_7 => 7,
    single_shot_rank_8 => 8,
    single_shot_rank_9 => 9,
    single_shot_rank_15 => 15,
    single_shot_rank_16 => 16,
    single_shot_rank_17 => 17,
    single_shot_rank_63 => 63,
    single_shot_rank_64 => 64,
    single_shot_rank_65 => 65,
}

/// Shots for the stabilizer-rank sampler on a dynamic circuit, which walks its
/// branch set once per shot.
const BRANCH_SHOTS: usize = 400;
/// Standard errors a single outcome frequency may sit from its exact weight.
const SIGMAS: f64 = 5.0;
/// Slack added to every frequency band, so a near-deterministic outcome never
/// demands an exact hit.
const FLOOR: f64 = 0.002;
const SUPPORT_EPS: f64 = 1e-12;

const CLIFFORD_1Q: [Gate; 7] = [
    Gate::H,
    Gate::S,
    Gate::Sdg,
    Gate::X,
    Gate::Y,
    Gate::Z,
    Gate::SX,
];
const CLIFFORD_T_1Q: [Gate; 9] = [
    Gate::H,
    Gate::S,
    Gate::Sdg,
    Gate::X,
    Gate::Y,
    Gate::Z,
    Gate::SX,
    Gate::T,
    Gate::Tdg,
];

type Counts = HashMap<Vec<u64>, u64>;
type Dist = HashMap<Vec<u64>, f64>;

#[derive(Clone, Copy)]
enum OneQubit {
    Clifford,
    CliffordT,
    Rotations,
}

fn random_1q(rng: &mut ChaCha8Rng, set: OneQubit) -> Gate {
    match set {
        OneQubit::Clifford => CLIFFORD_1Q[rng.random_range(0..CLIFFORD_1Q.len())].clone(),
        OneQubit::CliffordT => CLIFFORD_T_1Q[rng.random_range(0..CLIFFORD_T_1Q.len())].clone(),
        OneQubit::Rotations => {
            let theta = rng.random_range(0.1..3.0);
            match rng.random_range(0..4) {
                0 => Gate::H,
                1 => Gate::Rx(theta),
                2 => Gate::Ry(theta),
                _ => Gate::Rz(theta),
            }
        }
    }
}

/// One random 1q gate per qubit of `qubits`, then `qubits.len() / 2` random
/// CX or CZ pairs drawn within `qubits`.
fn random_layer(c: &mut Circuit, qubits: &[usize], set: OneQubit, rng: &mut ChaCha8Rng) {
    for &q in qubits {
        c.add_gate(random_1q(rng, set), &[q]);
    }
    for _ in 0..qubits.len() / 2 {
        let a = rng.random_range(0..qubits.len());
        let b = (a + 1 + rng.random_range(0..qubits.len() - 1)) % qubits.len();
        let gate = if rng.random_bool(0.5) {
            Gate::Cx
        } else {
            Gate::Cz
        };
        c.add_gate(gate, &[qubits[a], qubits[b]]);
    }
}

fn shuffled(len: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut v: Vec<usize> = (0..len).collect();
    for i in (1..len).rev() {
        v.swap(i, rng.random_range(0..=i));
    }
    v
}

/// Measure five qubits into five scattered bits, then a sixth qubit into the
/// first of those bits again, so the register carries a permuted map, unused
/// bits, and one bit whose last write wins.
fn measure_scrambled(c: &mut Circuit, qubits: &[usize], rng: &mut ChaCha8Rng) {
    let order = shuffled(qubits.len(), rng);
    let bits = shuffled(c.num_classical_bits, rng);
    for k in 0..5 {
        c.add_measure(qubits[order[k]], bits[k]);
    }
    c.add_measure(qubits[order[5]], bits[0]);
}

fn terminal_case(set: OneQubit, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let qubits: Vec<usize> = (0..6).collect();
    let mut c = Circuit::new(6, 9);
    for _ in 0..3 {
        random_layer(&mut c, &qubits, set, &mut rng);
    }
    measure_scrambled(&mut c, &qubits, &mut rng);
    c
}

/// Mid-circuit measurements into two bits, then another layer and terminal
/// measurements that overwrite the first mid-circuit bit. Without
/// conditionals both measured qubits are reset, the form the deferred compiled
/// sampler accepts. With them only the first is, and gates conditioned on both
/// bits follow.
fn dynamic_case(n: usize, set: OneQubit, conditionals: bool, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let qubits: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n, 8);
    let order = shuffled(n, &mut rng);
    let bits = shuffled(8, &mut rng);
    random_layer(&mut c, &qubits, set, &mut rng);
    random_layer(&mut c, &qubits, set, &mut rng);
    c.add_measure(order[0], bits[0]);
    c.add_measure(order[1], bits[1]);
    c.add_reset(order[0]);
    if !conditionals {
        c.add_reset(order[1]);
    }
    if conditionals {
        push_conditional(
            &mut c,
            ClassicalCondition::BitIsOne(bits[0]),
            Gate::X,
            order[2],
        );
        push_conditional(
            &mut c,
            ClassicalCondition::BitIsZero(bits[1]),
            Gate::X,
            order[0],
        );
        push_conditional(
            &mut c,
            ClassicalCondition::BitIsOne(bits[1]),
            Gate::Z,
            order[3],
        );
    }
    random_layer(&mut c, &qubits, set, &mut rng);
    c.add_measure(order[2], bits[2]);
    c.add_measure(order[3], bits[3]);
    c.add_measure(order[0], bits[4]);
    c.add_measure(order[4], bits[0]);
    c
}

fn push_conditional(c: &mut Circuit, condition: ClassicalCondition, gate: Gate, qubit: usize) {
    c.instructions.push(Instruction::Conditional {
        condition,
        gate,
        targets: SmallVec::from_slice(&[qubit]),
    });
}

/// `X q0; measure q0 -> c3; reset q0`, then `X q1`, conditioned on `c3` when
/// `conditional` is set, `X q2`, and terminal reads that overwrite `c3` with
/// the reset qubit. Every route must return `c0 = c1 = 1` and nothing else.
fn deterministic_dynamic_case(conditional: bool) -> Circuit {
    let mut c = Circuit::new(3, 5);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 3);
    c.add_reset(0);
    if conditional {
        push_conditional(&mut c, ClassicalCondition::BitIsOne(3), Gate::X, 1);
    } else {
        c.add_gate(Gate::X, &[1]);
    }
    c.add_gate(Gate::X, &[2]);
    c.add_measure(1, 0);
    c.add_measure(2, 1);
    c.add_measure(0, 3);
    c
}

fn key_words(num_bits: usize) -> usize {
    num_bits.div_ceil(64).max(1)
}

fn write_bit(key: &mut [u64], bit: usize, value: bool) {
    let mask = 1u64 << (bit % 64);
    if value {
        key[bit / 64] |= mask;
    } else {
        key[bit / 64] &= !mask;
    }
}

fn read_bit(key: &[u64], bit: usize) -> bool {
    (key[bit / 64] >> (bit % 64)) & 1 == 1
}

/// Exact distribution over classical keys. A terminal circuit reads the
/// measured qubits off the dense output distribution in program order. A
/// dynamic circuit is first rewritten into its deferred-measurement form:
/// each measurement copies its qubit onto a fresh wire, a reset moves the
/// qubit onto a fresh wire, and a conditional gate becomes a gate controlled
/// on the wire that last wrote its bit.
fn exact_distribution(circuit: &Circuit) -> Dist {
    let words = key_words(circuit.num_classical_bits);
    let mut dist = Dist::new();
    if circuit.has_terminal_measurements_only() && !circuit.has_resets() {
        let probs = sv_reference_probs(&circuit.without_measurements());
        let map = circuit.measurement_map();
        for (index, &p) in probs.iter().enumerate() {
            if p < SUPPORT_EPS {
                continue;
            }
            let mut key = vec![0u64; words];
            for &(qubit, bit) in &map {
                write_bit(&mut key, bit, (index >> qubit) & 1 == 1);
            }
            *dist.entry(key).or_insert(0.0) += p;
        }
        return dist;
    }

    let extra = circuit
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Measure { .. } | Instruction::Reset { .. }))
        .count();
    let mut unitary = Circuit::new(circuit.num_qubits + extra, 0);
    let mut wire: Vec<usize> = (0..circuit.num_qubits).collect();
    let mut bit_wire: Vec<Option<usize>> = vec![None; circuit.num_classical_bits];
    let mut next = circuit.num_qubits;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                let mapped: Vec<usize> = targets.iter().map(|&q| wire[q]).collect();
                unitary.add_gate(gate.clone(), &mapped);
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                unitary.add_gate(Gate::Cx, &[wire[*qubit], next]);
                bit_wire[*classical_bit] = Some(next);
                next += 1;
            }
            Instruction::Reset { qubit } => {
                wire[*qubit] = next;
                next += 1;
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                let (bit, on_one) = match condition {
                    ClassicalCondition::BitIsOne(b) => (*b, true),
                    ClassicalCondition::BitIsZero(b) => (*b, false),
                    other => panic!("no deferred form for {other:?}"),
                };
                let target = wire[targets[0]];
                let controlled = match gate {
                    Gate::X => Gate::Cx,
                    Gate::Z => Gate::Cz,
                    other => panic!("no controlled form for {other:?}"),
                };
                match bit_wire[bit] {
                    None if !on_one => unitary.add_gate(gate.clone(), &[target]),
                    None => {}
                    Some(control) => {
                        if !on_one {
                            unitary.add_gate(Gate::X, &[control]);
                        }
                        unitary.add_gate(controlled, &[control, target]);
                        if !on_one {
                            unitary.add_gate(Gate::X, &[control]);
                        }
                    }
                }
            }
            Instruction::Barrier { .. } => {}
            other => panic!("no deferred form for {other:?}"),
        }
    }
    let probs = sv_reference_probs(&unitary);
    for (index, &p) in probs.iter().enumerate() {
        if p < SUPPORT_EPS {
            continue;
        }
        let mut key = vec![0u64; words];
        for (bit, w) in bit_wire.iter().enumerate() {
            if let Some(w) = w {
                write_bit(&mut key, bit, (index >> w) & 1 == 1);
            }
        }
        *dist.entry(key).or_insert(0.0) += p;
    }
    dist
}

/// Check `counts` against the exact distribution: total, key width, support,
/// exact equality when the outcome is deterministic, a per-outcome band of
/// [`SIGMAS`] binomial standard errors, and a total variation bound of the
/// summed standard errors plus [`SIGMAS`] times the widest one.
fn assert_matches(label: &str, exact: &Dist, counts: &Counts, num_bits: usize, shots: usize) {
    let words = key_words(num_bits);
    let total: u64 = counts.values().sum();
    assert_eq!(total, shots as u64, "{label}: counts sum to {total}");
    for key in counts.keys() {
        assert_eq!(
            key.len(),
            words,
            "{label}: key {key:?} is not {words} words"
        );
        for bit in num_bits..words * 64 {
            assert!(
                !read_bit(key, bit),
                "{label}: key {key:?} sets padding bit {bit}"
            );
        }
        assert!(
            exact.contains_key(key),
            "{label}: sampled key {key:?} lies outside the exact support"
        );
    }
    if exact.len() == 1 {
        let (key, _) = exact.iter().next().unwrap();
        let expected: Counts = [(key.clone(), shots as u64)].into();
        assert_eq!(counts, &expected, "{label}: deterministic outcome");
        return;
    }
    let n = shots as f64;
    let mut tvd = 0.0;
    let mut sigma_sum = 0.0;
    let mut sigma_max: f64 = 0.0;
    for (key, &p) in exact {
        let f = counts.get(key).copied().unwrap_or(0) as f64 / n;
        let sigma = (p * (1.0 - p) / n).sqrt();
        assert!(
            (f - p).abs() <= SIGMAS * sigma + FLOOR,
            "{label}: key {key:?} frequency {f:.5} vs exact {p:.5} (sigma {sigma:.5})"
        );
        tvd += 0.5 * (f - p).abs();
        sigma_sum += sigma;
        sigma_max = sigma_max.max(sigma);
    }
    let bound = sigma_sum + SIGMAS * sigma_max;
    assert!(
        tvd <= bound,
        "{label}: total variation {tvd:.5} exceeds {bound:.5}"
    );
}

/// A `simulate` route, pinned by the provenance its result must carry so a
/// routing change cannot quietly collapse two routes into one. A backend's
/// terminal sampler and its per-shot replay share a label; the circuit's shape
/// decides which one runs.
struct Route {
    label: &'static str,
    kind: BackendKind,
    noisy: bool,
    backend: ResolvedBackend,
    engine: Option<Engine>,
}

fn route(label: &'static str, kind: BackendKind, backend: ResolvedBackend) -> Route {
    Route {
        label,
        kind,
        noisy: false,
        backend,
        engine: None,
    }
}

fn compiled(label: &'static str, kind: BackendKind) -> Route {
    Route {
        engine: Some(Engine::CompiledSampler),
        ..route(label, kind, ResolvedBackend::CompiledStabilizer)
    }
}

/// Zero-rate depolarizing noise, which keeps the distribution and sends the
/// circuit through the noisy entry point's own routing.
fn noisy(label: &'static str, backend: ResolvedBackend, engine: Option<Engine>) -> Route {
    Route {
        noisy: true,
        engine,
        ..route(label, BackendKind::Auto, backend)
    }
}

fn mps() -> BackendKind {
    BackendKind::Mps { max_bond_dim: 64 }
}

fn assert_provenance(label: &str, metadata: &RunMetadata, route: &Route) {
    assert_eq!(
        metadata.backend, route.backend,
        "{label}: resolved backend {:?}",
        metadata.backend
    );
    assert_eq!(metadata.engine, route.engine, "{label}: engine");
}

fn check_routes(case: &str, circuit: &Circuit, routes: &[Route], shots: usize) {
    check_routes_against(case, circuit, &exact_distribution(circuit), routes, shots);
}

fn check_routes_against(
    case: &str,
    circuit: &Circuit,
    exact: &Dist,
    routes: &[Route],
    shots: usize,
) {
    let bits = circuit.num_classical_bits;
    let zero_noise = NoiseModel::uniform_depolarizing(circuit, 0.0);
    for route in routes {
        let label = format!("{case} / {}", route.label);
        let base = || {
            let sim = simulate(circuit).backend(route.kind.clone()).seed(SEED);
            if route.noisy {
                sim.noise(&zero_noise)
            } else {
                sim
            }
        };
        let result = base()
            .shots(shots)
            .unwrap_or_else(|e| panic!("{label}: shots failed: {e}"));
        assert_provenance(&format!("{label} shots"), &result.metadata, route);
        assert_eq!(result.num_shots(), shots, "{label}: shot count");
        assert_eq!(result.num_classical_bits(), bits, "{label}: register width");
        assert!(
            result.shots.iter().all(|s| s.len() == bits),
            "{label}: a shot does not span the register"
        );
        assert_matches(
            &format!("{label} shots"),
            exact,
            &result.counts(),
            bits,
            shots,
        );

        let counts = base()
            .sample_counts(shots)
            .unwrap_or_else(|e| panic!("{label}: sample_counts failed: {e}"));
        assert_provenance(&format!("{label} counts"), &counts.metadata, route);
        assert_eq!(
            counts.num_classical_bits, bits,
            "{label}: counts register width"
        );
        assert_matches(
            &format!("{label} counts"),
            exact,
            &counts.counts,
            bits,
            shots,
        );
    }
}

/// The two compiled entry points that sit outside `simulate`:
/// `run_shots_compiled`, and `CompiledSampler::sample_counts`, whose keys are
/// in measurement-record order and are re-keyed here through the map.
fn check_compiled_entry_points(case: &str, circuit: &Circuit, shots: usize) {
    let exact = exact_distribution(circuit);
    let bits = circuit.num_classical_bits;

    let result = run_shots_compiled(circuit, shots, SEED).unwrap();
    assert_eq!(result.metadata.engine, Some(Engine::CompiledSampler));
    assert!(
        result.shots.iter().all(|s| s.len() == bits),
        "{case} / run_shots_compiled: a shot does not span the register"
    );
    assert_matches(
        &format!("{case} / run_shots_compiled"),
        &exact,
        &result.counts(),
        bits,
        shots,
    );

    let map = circuit.measurement_map();
    let mut sampler = compile_measurements(circuit, SEED).unwrap();
    let mut rekeyed = Counts::new();
    for (record, count) in sampler.sample_counts(shots) {
        let mut key = vec![0u64; key_words(bits)];
        for (index, &(_, bit)) in map.iter().enumerate() {
            write_bit(&mut key, bit, read_bit(&record, index));
        }
        *rekeyed.entry(key).or_insert(0) += count;
    }
    assert_matches(
        &format!("{case} / compiled record counts"),
        &exact,
        &rekeyed,
        bits,
        shots,
    );
}

#[test]
fn clifford_terminal_routes_agree() {
    use ResolvedBackend as R;
    for seed in [SEED, SEED + 1, SEED + 2] {
        let circuit = terminal_case(OneQubit::Clifford, seed);
        let case = format!("clifford terminal seed {seed}");
        check_routes(
            &case,
            &circuit,
            &[
                compiled("auto", BackendKind::Auto),
                compiled("stabilizer", BackendKind::Stabilizer),
                route("statevector", BackendKind::Statevector, R::Statevector),
                route("sparse", BackendKind::Sparse, R::Sparse),
                route("factored", BackendKind::Factored, R::Factored),
                route("mps", mps(), R::Mps),
                route(
                    "tensor network",
                    BackendKind::TensorNetwork,
                    R::TensorNetwork,
                ),
                route(
                    "density matrix",
                    BackendKind::DensityMatrix,
                    R::DensityMatrix,
                ),
                noisy(
                    "zero noise",
                    R::CompiledStabilizer,
                    Some(Engine::NoisyCompiledSampler),
                ),
            ],
            SHOTS,
        );
        check_compiled_entry_points(&case, &circuit, SHOTS);
    }
}

#[test]
fn clifford_t_terminal_routes_agree() {
    use ResolvedBackend as R;
    for seed in [SEED, SEED + 1] {
        let circuit = terminal_case(OneQubit::CliffordT, seed);
        assert!(circuit.has_t_gates());
        check_routes(
            &format!("clifford+t terminal seed {seed}"),
            &circuit,
            &[
                route("auto", BackendKind::Auto, R::Statevector),
                route("statevector", BackendKind::Statevector, R::Statevector),
                route(
                    "stabilizer rank",
                    BackendKind::StabilizerRank,
                    R::StabilizerRank,
                ),
                route("sparse", BackendKind::Sparse, R::Sparse),
                route("factored", BackendKind::Factored, R::Factored),
                route("mps", mps(), R::Mps),
                route(
                    "density matrix",
                    BackendKind::DensityMatrix,
                    R::DensityMatrix,
                ),
                noisy("zero noise", R::Statevector, None),
            ],
            SHOTS,
        );
    }
}

#[test]
fn rotation_terminal_routes_agree() {
    use ResolvedBackend as R;
    let circuit = terminal_case(OneQubit::Rotations, SEED);
    check_routes(
        "rotation terminal",
        &circuit,
        &[
            route("auto", BackendKind::Auto, R::Statevector),
            route("sparse", BackendKind::Sparse, R::Sparse),
            route("factored", BackendKind::Factored, R::Factored),
            route("mps", mps(), R::Mps),
            route(
                "tensor network",
                BackendKind::TensorNetwork,
                R::TensorNetwork,
            ),
            route(
                "density matrix",
                BackendKind::DensityMatrix,
                R::DensityMatrix,
            ),
            noisy("zero noise", R::Statevector, None),
        ],
        SHOTS,
    );
}

#[test]
fn product_and_decomposed_terminal_routes_agree() {
    use ResolvedBackend as R;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut product = Circuit::new(6, 9);
    for q in 0..6 {
        product.add_gate(random_1q(&mut rng, OneQubit::Rotations), &[q]);
        product.add_gate(random_1q(&mut rng, OneQubit::Rotations), &[q]);
    }
    measure_scrambled(&mut product, &[0, 1, 2, 3, 4, 5], &mut rng);
    check_routes(
        "product terminal",
        &product,
        &[
            route("auto", BackendKind::Auto, R::ProductState),
            route("product", BackendKind::ProductState, R::ProductState),
            route("statevector", BackendKind::Statevector, R::Statevector),
        ],
        SHOTS,
    );

    let mut blocks = Circuit::new(9, 9);
    for block in [&[0usize, 1, 2, 3][..], &[4, 5, 6], &[7, 8]] {
        for _ in 0..3 {
            random_layer(&mut blocks, block, OneQubit::Rotations, &mut rng);
        }
    }
    measure_scrambled(&mut blocks, &[0, 4, 7, 2, 5, 8], &mut rng);
    check_routes(
        "decomposed terminal",
        &blocks,
        &[
            route("auto", BackendKind::Auto, R::Decomposed),
            route("statevector", BackendKind::Statevector, R::Statevector),
        ],
        SHOTS,
    );
}

#[test]
fn clifford_register_route_agrees() {
    use ResolvedBackend as R;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let qubits: Vec<usize> = (0..16).collect();
    let mut c = Circuit::new(16, 9);
    random_layer(&mut c, &qubits, OneQubit::Clifford, &mut rng);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::T, &[9]);
    random_layer(&mut c, &qubits, OneQubit::Clifford, &mut rng);
    c.add_gate(Gate::Tdg, &[11]);
    random_layer(&mut c, &qubits, OneQubit::Clifford, &mut rng);
    measure_scrambled(&mut c, &qubits, &mut rng);
    check_routes(
        "clifford register terminal",
        &c,
        &[
            route("auto", BackendKind::Auto, R::StabilizerRank),
            route("statevector", BackendKind::Statevector, R::Statevector),
        ],
        SHOTS,
    );
}

#[test]
fn clifford_dynamic_routes_agree() {
    use ResolvedBackend as R;
    let circuit = dynamic_case(5, OneQubit::Clifford, false, SEED);
    check_routes(
        "clifford dynamic",
        &circuit,
        &[
            compiled("auto deferred", BackendKind::Auto),
            route("statevector", BackendKind::Statevector, R::Statevector),
            route("sparse", BackendKind::Sparse, R::Sparse),
            route("mps", mps(), R::Mps),
            noisy("zero noise", R::Stabilizer, None),
        ],
        SHOTS,
    );

    let circuit = dynamic_case(5, OneQubit::Clifford, true, SEED);
    check_routes(
        "clifford conditional",
        &circuit,
        &[
            route("auto", BackendKind::Auto, R::Stabilizer),
            route("statevector", BackendKind::Statevector, R::Statevector),
            route("factored", BackendKind::Factored, R::Factored),
            route("mps", mps(), R::Mps),
        ],
        SHOTS,
    );
}

#[test]
fn non_clifford_dynamic_routes_agree() {
    use ResolvedBackend as R;
    let circuit = dynamic_case(5, OneQubit::CliffordT, true, SEED);
    assert!(circuit.has_t_gates());
    check_routes(
        "clifford+t conditional",
        &circuit,
        &[
            route("auto", BackendKind::Auto, R::Statevector),
            route("sparse", BackendKind::Sparse, R::Sparse),
            route("mps", mps(), R::Mps),
            noisy("zero noise", R::Statevector, None),
        ],
        SHOTS,
    );
    check_routes(
        "clifford+t conditional",
        &circuit,
        &[route(
            "stabilizer rank",
            BackendKind::StabilizerRank,
            R::StabilizerRank,
        )],
        BRANCH_SHOTS,
    );

    let circuit = dynamic_case(5, OneQubit::Rotations, true, SEED);
    check_routes(
        "rotation conditional",
        &circuit,
        &[
            route("auto", BackendKind::Auto, R::Statevector),
            route("factored", BackendKind::Factored, R::Factored),
            route("mps", mps(), R::Mps),
            noisy("zero noise", R::Statevector, None),
        ],
        SHOTS,
    );

    // Ten qubits leave a stabilizer-rank budget of two T gates, which Auto
    // spends on the branch sampler instead of per-shot statevector runs.
    let mut circuit = Circuit::new(10, 8);
    for q in [0, 1] {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit
        .instructions
        .extend(dynamic_case(10, OneQubit::Clifford, true, SEED).instructions);
    assert_eq!(circuit.t_count(), 2);
    check_routes(
        "clifford+t auto stabilizer rank",
        &circuit,
        &[route("auto", BackendKind::Auto, R::StabilizerRank)],
        BRANCH_SHOTS,
    );
}

#[test]
fn deterministic_dynamic_routes_return_one_key() {
    use ResolvedBackend as R;
    let expected: Dist = [(vec![0b00011u64], 1.0)].into();
    for conditional in [false, true] {
        let circuit = deterministic_dynamic_case(conditional);
        assert_eq!(exact_distribution(&circuit), expected);
        let auto = if conditional {
            route("auto", BackendKind::Auto, R::ProductState)
        } else {
            compiled("auto deferred", BackendKind::Auto)
        };
        check_routes(
            &format!("deterministic dynamic, conditional {conditional}"),
            &circuit,
            &[
                auto,
                route("statevector", BackendKind::Statevector, R::Statevector),
                route("sparse", BackendKind::Sparse, R::Sparse),
                route("mps", mps(), R::Mps),
                route(
                    "density matrix",
                    BackendKind::DensityMatrix,
                    R::DensityMatrix,
                ),
            ],
            256,
        );
    }
}

const WIDE_QUBITS: usize = 72;
const WIDE_BITS: usize = 80;

/// GHZ groups on scattered qubits of a 72-qubit register read into 80 bits
/// through an injective scramble, with one bit overwritten by a later
/// measurement. With `dynamic`, a member of the first group is also read
/// mid-circuit into a spare bit, reset, and flipped.
fn wide_case(dynamic: bool) -> Circuit {
    let mut c = Circuit::new(WIDE_QUBITS, WIDE_BITS);
    for group in [&[0usize, 20, 41, 70][..], &[5, 33, 64], &[12, 50]] {
        c.add_gate(Gate::H, &[group[0]]);
        for pair in group.windows(2) {
            c.add_gate(Gate::Cx, &[pair[0], pair[1]]);
        }
    }
    for q in [3, 66, 71] {
        c.add_gate(Gate::X, &[q]);
    }
    c.add_gate(Gate::S, &[0]);
    if dynamic {
        c.add_measure(20, wide_spare_bit());
        c.add_reset(20);
        c.add_gate(Gate::X, &[20]);
    }
    for q in 0..WIDE_QUBITS {
        c.add_measure(q, wide_bit(q));
    }
    c.add_measure(3, wide_bit(70));
    c
}

fn wide_bit(qubit: usize) -> usize {
    (37 * qubit + 11) % WIDE_BITS
}

fn wide_spare_bit() -> usize {
    (0..WIDE_BITS)
        .rev()
        .find(|&b| (0..WIDE_QUBITS).all(|q| wide_bit(q) != b))
        .unwrap()
}

/// Exact distribution of [`wide_case`]: every `H` lands on a fresh qubit and
/// nothing after it mixes amplitudes, so each of the eight branches is a
/// classical bit string run through the instruction list.
fn wide_distribution(circuit: &Circuit) -> Dist {
    let leaders = [0usize, 5, 12];
    let mut dist = Dist::new();
    for branch in 0..8usize {
        let mut value = vec![false; circuit.num_qubits];
        let mut key = vec![0u64; key_words(circuit.num_classical_bits)];
        for inst in &circuit.instructions {
            match inst {
                Instruction::Gate { gate, targets } => match gate {
                    Gate::H => {
                        let g = leaders.iter().position(|&l| l == targets[0]).unwrap();
                        value[targets[0]] = (branch >> g) & 1 == 1;
                    }
                    Gate::Cx => value[targets[1]] ^= value[targets[0]],
                    Gate::X => value[targets[0]] ^= true,
                    Gate::S => {}
                    other => panic!("unexpected gate {other:?}"),
                },
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => write_bit(&mut key, *classical_bit, value[*qubit]),
                Instruction::Reset { qubit } => value[*qubit] = false,
                other => panic!("unexpected instruction {other:?}"),
            }
        }
        *dist.entry(key).or_insert(0.0) += 0.125;
    }
    dist
}

#[test]
fn wide_register_routes_agree() {
    use ResolvedBackend as R;
    let circuit = wide_case(false);
    let exact = wide_distribution(&circuit);
    assert_eq!(exact.len(), 8);
    assert!(exact.keys().all(|k| k.len() == 2 && k[1] != 0));

    let routes = [
        compiled("auto", BackendKind::Auto),
        route("mps, split into blocks", mps(), R::Decomposed),
        noisy(
            "zero noise",
            R::CompiledStabilizer,
            Some(Engine::FrameSampler),
        ),
    ];
    check_routes_against("wide terminal", &circuit, &exact, &routes, SHOTS);

    let result = run_shots_compiled(&circuit, SHOTS, SEED).unwrap();
    assert_matches(
        "wide terminal / run_shots_compiled",
        &exact,
        &result.counts(),
        WIDE_BITS,
        SHOTS,
    );

    let circuit = wide_case(true);
    let exact = wide_distribution(&circuit);
    check_routes_against(
        "wide dynamic",
        &circuit,
        &exact,
        &[compiled("auto deferred", BackendKind::Auto)],
        SHOTS,
    );
}

fn parity(bits: impl Fn(usize) -> bool, row: &[usize]) -> bool {
    row.iter().fold(false, |acc, &m| acc ^ bits(m))
}

/// Distribution of the parity pattern `rows` projects out of `exact`, reading
/// measurement `m` from classical bit `m`.
fn parity_distribution(exact: &Dist, rows: &[Vec<usize>]) -> Dist {
    let mut out = Dist::new();
    for (key, &p) in exact {
        let mut pattern = vec![0u64];
        for (e, row) in rows.iter().enumerate() {
            write_bit(&mut pattern, e, parity(|m| read_bit(key, m), row));
        }
        *out.entry(pattern).or_insert(0.0) += p;
    }
    out
}

fn pattern_counts(packed: &PackedShots, num_rows: usize) -> Counts {
    let mut counts = Counts::new();
    for shot in 0..packed.num_shots() {
        let mut pattern = vec![0u64];
        for e in 0..num_rows {
            write_bit(&mut pattern, e, packed.get_bit(shot, e));
        }
        *counts.entry(pattern).or_insert(0) += 1;
    }
    counts
}

/// Measurement `m` writes classical bit `m`, so record parities and register
/// parities are the same function of one shot.
fn detector_case(dynamic: bool) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut c = Circuit::new(6, 9);
    random_layer(&mut c, &[0, 1, 2, 3], OneQubit::Clifford, &mut rng);
    c.add_gate(Gate::Cx, &[0, 5]);
    c.add_gate(Gate::X, &[4]);
    let mut bit = 0;
    if dynamic {
        c.add_measure(1, bit);
        c.add_reset(1);
        bit += 1;
        c.add_measure(2, bit);
        c.add_reset(2);
        bit += 1;
        random_layer(&mut c, &[1, 2, 3], OneQubit::Clifford, &mut rng);
    }
    for q in [0, 5, 4, 1, 2, 3] {
        c.add_measure(q, bit);
        bit += 1;
    }
    c
}

#[test]
fn detection_events_match_record_parity() {
    let circuit = detector_case(false);
    let exact = exact_distribution(&circuit);
    // Records 0 and 1 read q0 and its copy q5, record 2 reads the flipped q4.
    let pairs = [(0usize, 1usize), (2, 3), (1, 4), (3, 5)];
    let rows: Vec<Vec<usize>> = pairs.iter().map(|&(a, b)| vec![a, b]).collect();
    let want = parity_distribution(&exact, &rows);
    assert_eq!(want.keys().filter(|k| k[0] & 1 == 1).count(), 0);

    let mut sampler = compile_measurements(&circuit, SEED).unwrap();
    let events = sampler.sample_detection_events(&pairs, SHOTS);
    assert_matches(
        "detection events",
        &want,
        &pattern_counts(&events, pairs.len()),
        pairs.len(),
        SHOTS,
    );

    // Repeating a pair pushes the projected weight past the measurement
    // weight, which selects the path that samples records first.
    let repeated: Vec<(usize, usize)> = pairs.iter().copied().cycle().take(64).collect();
    let mut sampler = compile_measurements(&circuit, SEED).unwrap();
    let events = sampler.sample_detection_events(&repeated, SHOTS);
    for shot in 0..SHOTS {
        for e in pairs.len()..repeated.len() {
            assert_eq!(
                events.get_bit(shot, e),
                events.get_bit(shot, e % pairs.len()),
                "repeated pair {e} disagrees with its first copy in shot {shot}"
            );
        }
    }
    assert_matches(
        "detection events, record path",
        &want,
        &pattern_counts(&events, pairs.len()),
        pairs.len(),
        SHOTS,
    );
}

#[test]
fn detector_sampler_matches_record_parity() {
    for dynamic in [false, true] {
        let circuit = detector_case(dynamic);
        let exact = exact_distribution(&circuit);
        let num_records = circuit
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Measure { .. }))
            .count();
        let detectors: Vec<Vec<usize>> = vec![
            vec![0, 1],
            vec![1, 2, 3],
            vec![num_records - 1, num_records - 2],
            vec![2],
        ];
        let observables: Vec<Vec<usize>> = vec![vec![0, num_records - 1], vec![3, 4, 5]];
        let label = format!("detector sampler, dynamic {dynamic}");

        let mut sampler =
            compile_detector_sampler(&circuit, detectors.clone(), observables.clone(), SEED)
                .unwrap();
        assert_eq!(sampler.num_measurements(), num_records);
        let batch = sampler.sample_packed(SHOTS).unwrap();
        for shot in 0..SHOTS {
            let record = |m: usize| batch.measurements.get_bit(shot, m);
            for (d, row) in detectors.iter().enumerate() {
                assert_eq!(
                    batch.detectors.get_bit(shot, d),
                    parity(record, row),
                    "{label}: detector {d} in shot {shot}"
                );
            }
            for (o, row) in observables.iter().enumerate() {
                assert_eq!(
                    batch.observables.get_bit(shot, o),
                    parity(record, row),
                    "{label}: observable {o} in shot {shot}"
                );
            }
        }

        let all_records: Vec<Vec<usize>> = (0..num_records).map(|m| vec![m]).collect();
        assert_matches(
            &format!("{label} records"),
            &parity_distribution(&exact, &all_records),
            &pattern_counts(&batch.measurements, num_records),
            num_records,
            SHOTS,
        );
        assert_matches(
            &format!("{label} detectors"),
            &parity_distribution(&exact, &detectors),
            &pattern_counts(&batch.detectors, detectors.len()),
            detectors.len(),
            SHOTS,
        );
        assert_matches(
            &format!("{label} observables"),
            &parity_distribution(&exact, &observables),
            &pattern_counts(&batch.observables, observables.len()),
            observables.len(),
            SHOTS,
        );
    }
}
