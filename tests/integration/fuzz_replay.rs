//! Stable-toolchain replay of the fuzz targets in `fuzz/`: the checked-in seeds plus a
//! seeded sweep through the same decoders, so the checks run in the ordinary suite.

#[path = "../../fuzz/src/lib.rs"]
mod fuzz_checks;

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::Path;

use fuzz_checks::{
    FuzzQuery, SimulateCase, check_fusion, check_parse, check_parse_bytes, check_parse_tokens,
    check_simulate, check_simulate_case, simulate_case_from_bytes,
};
use prism_q::{
    BackendKind, Circuit, Gate, Instruction, Limits, NoiseChannel, NoiseEvent, NoiseModel, Query,
    ResolvedBackend, RouteRule, simulate,
};

fn seeds(target: &str) -> Vec<Vec<u8>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fuzz/seeds")
        .join(target);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut inputs: Vec<Vec<u8>> = entries
        .map(|entry| std::fs::read(entry.unwrap().path()).unwrap())
        .collect();
    inputs.sort();
    inputs
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

fn has_terminal_measurements_under_reset_channel(circuit: &Circuit) -> bool {
    let mut seen_measure = false;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Conditional { .. } | Instruction::Region(_) => return false,
            Instruction::Measure { .. } => seen_measure = true,
            Instruction::Gate { .. } if seen_measure => return false,
            Instruction::Reset { .. } if seen_measure => return false,
            _ => {}
        }
    }
    true
}

fn has_memory_noise(noise: &NoiseModel) -> bool {
    noise.after_gate.iter().flatten().any(|e| {
        matches!(
            e.channel,
            NoiseChannel::Leakage { .. }
                | NoiseChannel::Seepage { .. }
                | NoiseChannel::LeakageTransport { .. }
        ) || matches!(&e.channel, NoiseChannel::QuasiStatic { weights, .. } if !weights.is_empty())
    })
}

fn readout_is_inert(circuit: &Circuit, noise: &NoiseModel) -> bool {
    noise
        .readout
        .iter()
        .take(circuit.num_classical_bits)
        .all(|r| r.as_ref().is_none_or(|ro| ro.p01 == 0.0 && ro.p10 == 0.0))
}

fn dm_settle_defect(circuit: &Circuit, noise: Option<&NoiseModel>, query: &FuzzQuery) -> bool {
    let Some(noise) = noise else {
        return false;
    };
    if circuit.save_count() > 0
        || !has_terminal_measurements_under_reset_channel(circuit)
        || noise.validate_for(circuit).is_err()
        || has_memory_noise(noise)
    {
        return false;
    }
    if matches!(query, FuzzQuery::Run | FuzzQuery::Marginals) && !readout_is_inert(circuit, noise) {
        return false;
    }
    let mut first = vec![usize::MAX; circuit.num_qubits];
    let mut last = vec![usize::MAX; circuit.num_qubits];
    for (idx, inst) in circuit.instructions.iter().enumerate() {
        if let Instruction::Measure { qubit, .. } = *inst {
            first[qubit] = first[qubit].min(idx);
            last[qubit] = idx;
        }
    }
    let measured = |q: usize, idx: usize| first.get(q).is_some_and(|&m| m <= idx);
    let settled = |event: &NoiseEvent, idx: usize| {
        event
            .qubits
            .iter()
            .all(|&q| last.get(q).is_some_and(|&m| m <= idx))
    };
    noise.after_gate.iter().enumerate().any(|(idx, events)| {
        events
            .iter()
            .any(|event| !settled(event, idx) && event.qubits.iter().any(|&q| measured(q, idx)))
    })
}

fn is_unitary_no_saves(circuit: &Circuit) -> bool {
    circuit
        .instructions
        .iter()
        .all(|inst| matches!(inst, Instruction::Gate { .. } | Instruction::Barrier { .. }))
}

fn prob_cap_from_limits(limits: &Limits) -> Option<usize> {
    let from_budget = limits
        .memory_budget_bytes
        .map(|bytes| (bytes / 8).checked_ilog2().unwrap_or(0) as usize);
    match (limits.max_probability_qubits, from_budget) {
        (Some(explicit), _) => Some(explicit),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn plan_for_case(case: &SimulateCase, circuit: &Circuit) -> Option<prism_q::Plan> {
    let query = match &case.query {
        FuzzQuery::Run => Query::Run,
        FuzzQuery::Shots(n) => Query::Shots(*n),
        FuzzQuery::Counts(n) => Query::Counts(*n),
        FuzzQuery::Marginals => Query::Marginals,
        FuzzQuery::Expectation(obs) => Query::Expectation(obs.as_slice()),
    };
    let mut req = simulate(circuit)
        .backend(case.kind.clone())
        .limits(case.limits);
    if let Some(noise) = case.base_noise.as_ref() {
        let model;
        let bound = if circuit.num_qubits == case.base.num_qubits {
            noise
        } else {
            model = case.padded_noise.as_ref().expect("padded_noise");
            model
        };
        req = req.noise(bound);
    }
    if case.require_exact {
        req = req.require_exact();
    }
    req.plan(query).ok()
}

fn plan_succeeds(case: &SimulateCase, circuit: &Circuit) -> bool {
    plan_for_case(case, circuit).is_some_and(|p| p.would_fail.is_none())
}

fn stabilizer_rank_defect(case: &SimulateCase) -> bool {
    if !matches!(case.kind, BackendKind::StabilizerRank)
        || case.base_noise.is_some()
        || case.base.save_count() > 0
    {
        return false;
    }
    let check_one = |c: &Circuit| {
        let Some(plan) = plan_for_case(case, c) else {
            return false;
        };
        if plan.would_fail.is_some() {
            return false;
        }
        match &case.query {
            FuzzQuery::Run => {
                let cap_violated = plan.backend == ResolvedBackend::StabilizerRank
                    && case
                        .limits
                        .max_probability_qubits
                        .is_some_and(|cap| c.num_qubits > cap);
                let empty_bits =
                    plan.backend == ResolvedBackend::StabilizerRank && c.num_classical_bits > 0;
                let unsupported =
                    !is_unitary_no_saves(c) || !c.is_clifford_plus_t() || c.num_qubits > 25;
                cap_violated || empty_bits || unsupported
            }
            FuzzQuery::Marginals => !is_unitary_no_saves(c) || !c.is_clifford_plus_t(),
            FuzzQuery::Shots(_) | FuzzQuery::Counts(_) => {
                c.has_t_gates() && !c.is_clifford_plus_t()
            }
            FuzzQuery::Expectation(_) => false,
        }
    };
    check_one(&case.base)
        || (case.padded.num_qubits > case.base.num_qubits && check_one(&case.padded))
}

fn draws_randomness(circuit: &Circuit) -> bool {
    circuit.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
                | Instruction::Region(_)
        )
    })
}

fn batch_metadata_reason_defect(case: &SimulateCase) -> bool {
    if !case.check_batch
        || case.has_request_limits
        || case.require_exact
        || case.base_noise.is_some()
        || !matches!(case.query, FuzzQuery::Run)
    {
        return false;
    }
    let check_one = |c: &Circuit| -> Option<bool> {
        let plan = simulate(c)
            .backend(case.kind.clone())
            .plan(prism_q::Query::Run)
            .ok()?;
        if plan.would_fail.is_some() {
            return None;
        }
        let direct = !draws_randomness(c)
            && !matches!(
                plan.rule,
                RouteRule::IndependentBlocks
                    | RouteRule::TemporalCliffordPrefix
                    | RouteRule::LargeCliffordBlocks
            )
            && !matches!(
                plan.backend,
                ResolvedBackend::StabilizerRank
                    | ResolvedBackend::StochasticPauli
                    | ResolvedBackend::DeterministicPauli
                    | ResolvedBackend::PauliPath
            );
        Some(direct && plan.rule != RouteRule::Default)
    };
    let Some(base_hit) = check_one(&case.base) else {
        return false;
    };
    if case.padded.num_qubits > case.base.num_qubits {
        let Some(padded_hit) = check_one(&case.padded) else {
            return false;
        };
        base_hit || padded_hit
    } else {
        base_hit
    }
}

fn has_gate(circuit: &Circuit, pred: fn(&Gate) -> bool) -> bool {
    fn check(insts: &[Instruction], pred: fn(&Gate) -> bool) -> bool {
        insts.iter().any(|inst| match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => pred(gate),
            Instruction::Region(region) => check(region.body(), pred),
            _ => false,
        })
    }
    check(&circuit.instructions, pred)
}

fn is_unitary_like(circuit: &Circuit) -> bool {
    !circuit.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
                | Instruction::Region(_)
        )
    })
}

fn gate_outside_pauli_lowering(gate: &Gate) -> bool {
    match gate {
        Gate::Fused2q(_)
        | Gate::Unitary(_)
        | Gate::Mcu(_)
        | Gate::QftBlock { .. }
        | Gate::MultiFused(_)
        | Gate::Multi2q(_)
        | Gate::BatchRzz(_)
        | Gate::BatchPhase(_)
        | Gate::DiagonalBatch(_)
        | Gate::MultiPauliRot(_) => true,
        Gate::Cu(mat) => {
            let eps = 1e-12;
            let is_diag = mat[0][1].norm() <= eps && mat[1][0].norm() <= eps;
            let is_anti = mat[0][0].norm() <= eps && mat[1][1].norm() <= eps;
            !(is_diag || is_anti)
        }
        _ => false,
    }
}

fn pauli_shot_pershot_defect(case: &SimulateCase) -> bool {
    matches!(case.kind, BackendKind::PauliPath { .. })
        && matches!(case.query, FuzzQuery::Shots(_) | FuzzQuery::Counts(_))
        && case.base_noise.is_none()
        && case.base.save_count() == 0
        && !case
            .base
            .fold_static_guards()
            .has_terminal_measurements_only()
}

fn dm_prob_cap_defect(case: &SimulateCase) -> bool {
    if !matches!(case.kind, BackendKind::DensityMatrix) || !matches!(case.query, FuzzQuery::Run) {
        return false;
    }
    let Some(prob_cap) = case.limits.max_probability_qubits else {
        return false;
    };
    let check_one = |c: &Circuit| c.num_qubits > prob_cap && plan_succeeds(case, c);
    check_one(&case.base)
        || (case.padded.num_qubits > case.base.num_qubits && check_one(&case.padded))
}

fn sparse_dense_unitary_defect(case: &SimulateCase) -> bool {
    if !matches!(case.kind, BackendKind::Sparse)
        || case.base.save_count() > 0
        || !has_gate(&case.base, |g| matches!(g, Gate::Unitary(_)))
    {
        return false;
    }
    plan_succeeds(case, &case.base)
        || (case.padded.num_qubits > case.base.num_qubits && plan_succeeds(case, &case.padded))
}

fn pauli_unsupported_gate_defect(case: &SimulateCase) -> bool {
    if case.base.save_count() > 0
        || !is_unitary_like(&case.base)
        || !plan_succeeds(case, &case.base)
    {
        return false;
    }
    let matches_query = match (&case.kind, &case.query) {
        (
            BackendKind::StochasticPauli { .. } | BackendKind::DeterministicPauli { .. },
            FuzzQuery::Marginals | FuzzQuery::Expectation(_),
        ) => case.base_noise.is_none(),
        (BackendKind::PauliPath { .. }, FuzzQuery::Expectation(_)) => case
            .base_noise
            .as_ref()
            .is_none_or(|n| readout_is_inert(&case.base, n)),
        _ => false,
    };
    if !matches_query {
        return false;
    }
    let has_unsupported_noise = matches!(case.kind, BackendKind::PauliPath { .. })
        && case.base_noise.as_ref().is_some_and(|n| {
            has_memory_noise(n)
                || n.after_gate.iter().flatten().any(|e| {
                    matches!(
                        e.channel,
                        NoiseChannel::Custom { .. } | NoiseChannel::Kraus2q { .. }
                    )
                })
        });
    has_gate(&case.base, gate_outside_pauli_lowering) || has_unsupported_noise
}

fn prob_cap_plan_mismatch_defect(case: &SimulateCase) -> bool {
    if case.base_noise.is_some() || case.base.save_count() > 0 {
        return false;
    }
    let Some(prob_cap) = prob_cap_from_limits(&case.limits) else {
        return false;
    };
    let check_shots_plan = |c: &Circuit| {
        if c.num_qubits <= prob_cap {
            return false;
        }
        let folded = c.fold_static_guards();
        if !folded.has_terminal_measurements_only() || !folded.measurement_map().is_empty() {
            return false;
        }
        plan_for_case(case, c).is_some_and(|p| {
            p.would_fail.is_none()
                && (p.backend == ResolvedBackend::Statevector
                    || (p.backend == ResolvedBackend::Decomposed
                        && matches!(
                            case.kind,
                            BackendKind::Auto
                                | BackendKind::Statevector
                                | BackendKind::ProductState
                                | BackendKind::Sparse
                        )))
        })
    };
    let check_run_plan = |c: &Circuit| {
        c.num_qubits > prob_cap
            && plan_for_case(case, c).is_some_and(|p| {
                p.would_fail.is_none()
                    && (p.backend == ResolvedBackend::ProductState
                        || (p.backend == ResolvedBackend::Sparse
                            && c.num_qubits == case.base.num_qubits))
            })
    };
    match &case.query {
        FuzzQuery::Shots(_) | FuzzQuery::Counts(_) => {
            check_shots_plan(&case.base)
                || (case.padded.num_qubits > case.base.num_qubits && check_shots_plan(&case.padded))
        }
        FuzzQuery::Run => {
            check_run_plan(&case.base)
                || (case.padded.num_qubits > case.base.num_qubits && check_run_plan(&case.padded))
        }
        _ => false,
    }
}

#[cfg(feature = "distributed")]
fn distributed_sv_cap_defect(case: &SimulateCase) -> bool {
    if !matches!(case.kind, BackendKind::StatevectorDistributed { .. }) {
        return false;
    }
    let from_budget = case
        .limits
        .memory_budget_bytes
        .map(|b| (b / 16).checked_ilog2().unwrap_or(0) as usize);
    let sv_cap = match (case.limits.max_statevector_qubits, from_budget) {
        (Some(explicit), _) => explicit,
        (None, Some(b)) => b,
        (None, None) => return false,
    };
    let check_one = |c: &Circuit| c.num_qubits > sv_cap && plan_succeeds(case, c);
    check_one(&case.base)
        || (case.padded.num_qubits > case.base.num_qubits && check_one(&case.padded))
}

#[cfg(not(feature = "distributed"))]
fn distributed_sv_cap_defect(_case: &SimulateCase) -> bool {
    false
}

// Open defects the Stage 2 sweep reaches on the base branch. A case inside one of these
// shapes may fail; the seeds in `fuzz/seeds/simulate_known_defects` pin one failure each.
fn inside_known_defect(case: &SimulateCase) -> bool {
    let dm_defect = matches!(case.kind, BackendKind::DensityMatrix)
        && matches!(
            case.query,
            FuzzQuery::Run | FuzzQuery::Shots(_) | FuzzQuery::Counts(_) | FuzzQuery::Marginals
        )
        && (dm_settle_defect(&case.base, case.base_noise.as_ref(), &case.query)
            || (case.padded.num_qubits > case.base.num_qubits
                && dm_settle_defect(&case.padded, case.padded_noise.as_ref(), &case.query)));

    dm_defect
        || stabilizer_rank_defect(case)
        || batch_metadata_reason_defect(case)
        || pauli_shot_pershot_defect(case)
        || dm_prob_cap_defect(case)
        || sparse_dense_unitary_defect(case)
        || pauli_unsupported_gate_defect(case)
        || prob_cap_plan_mismatch_defect(case)
        || distributed_sv_cap_defect(case)
}

#[test]
fn qasm_seeds_and_their_truncations_parse_without_panic() {
    let corpus = seeds("qasm_parse");
    assert!(
        corpus.len() > 100,
        "seed corpus holds {} files",
        corpus.len()
    );
    for seed in &corpus {
        check_parse_bytes(seed);
        for cut in (0..seed.len()).step_by(5) {
            check_parse_bytes(&seed[..cut]);
        }
    }
}

#[test]
fn spliced_qasm_seeds_parse_without_panic() {
    let corpus = seeds("qasm_parse");
    assert!(!corpus.is_empty(), "qasm_parse seed corpus is empty");
    let mut rng = SplitMix(42);
    for _ in 0..3000 {
        let text = String::from_utf8_lossy(&corpus[rng.below(corpus.len())]).into_owned();
        let donor = String::from_utf8_lossy(&corpus[rng.below(corpus.len())]).into_owned();
        let at = floor_char_boundary(&text, rng.below(text.len() + 1));
        let from = floor_char_boundary(&donor, rng.below(donor.len() + 1));
        let to = floor_char_boundary(&donor, from + rng.below(donor.len() - from + 1));
        let mut spliced = String::with_capacity(text.len() + to - from);
        spliced.push_str(&text[..at]);
        spliced.push_str(&donor[from..to]);
        spliced.push_str(&text[at..]);
        check_parse(&spliced);
    }
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[test]
fn seeded_token_programs_parse_without_panic() {
    let corpus = seeds("qasm_tokens");
    assert!(!corpus.is_empty(), "qasm_tokens seed corpus is empty");
    for seed in corpus {
        check_parse_tokens(&seed);
    }
    let mut rng = SplitMix(42);
    for _ in 0..3000 {
        let len = rng.below(96);
        check_parse_tokens(&rng.bytes(len));
    }
}

#[test]
fn seeded_fusion_circuits_match_unfused() {
    let corpus = seeds("fusion");
    assert!(!corpus.is_empty(), "fusion seed corpus is empty");
    for seed in corpus {
        check_fusion(&seed);
    }
    let mut rng = SplitMix(42);
    for _ in 0..150 {
        let len = 64 + rng.below(448);
        check_fusion(&rng.bytes(len));
    }
}

#[test]
fn seeded_simulate_cases_match_the_reference() {
    let corpus = seeds("simulate");
    assert!(!corpus.is_empty(), "simulate seed corpus is empty");
    for seed in corpus {
        check_simulate(&seed);
    }
    let mut rng = SplitMix(42);
    let mut tolerated = 0usize;
    for _ in 0..1800 {
        let len = 12 + rng.below(120);
        let case = simulate_case_from_bytes(&rng.bytes(len));
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| check_simulate_case(&case))) {
            if !inside_known_defect(&case) {
                resume_unwind(panic);
            }
            tolerated += 1;
        }
    }
    assert!(
        tolerated > 0,
        "no case failed inside a known defect shape; remove `inside_known_defect`"
    );
}

#[test]
fn known_defect_seeds_still_fail() {
    let corpus = seeds("simulate_known_defects");
    assert_eq!(corpus.len(), 9, "one seed per pinned open defect");
    for seed in corpus {
        let case = simulate_case_from_bytes(&seed);
        assert!(
            inside_known_defect(&case),
            "seed lies outside every known defect shape\n{case:#?}"
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| check_simulate_case(&case))).is_err(),
            "a known defect no longer reproduces; move its seed to fuzz/seeds/simulate\n{case:#?}"
        );
    }
}

#[test]
fn bounded_tn_shot_seed_reports_nontrivial_fidelity_bound() {
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/simulate/bounded_tn_shot_bound"),
    )
    .expect("bounded_tn_shot_bound seed");
    let case = simulate_case_from_bytes(&bytes);
    let shots = match case.query {
        FuzzQuery::Shots(s) => s,
        other => panic!("expected Shots query, got {other:?}"),
    };
    let result = simulate(&case.base)
        .backend(case.kind)
        .limits(case.limits)
        .seed(42)
        .shots(shots)
        .expect("bounded TN shots");
    let bound = result
        .metadata
        .fidelity_lower_bound()
        .expect("bounded TN reports fidelity lower bound");
    assert!(
        bound < 1.0,
        "expected nontrivial truncation bound < 1.0, got {bound}"
    );
}

#[cfg(feature = "distributed")]
#[test]
fn distributed_serial_simulate_cases_match_the_reference() {
    use prism_q::DistributedContext;

    let mut rng = SplitMix(0x5EED_D157);
    for _ in 0..80 {
        let len = 16 + rng.below(100);
        let mut bytes = rng.bytes(len);
        // Force Stage 2 header without batch check and select index 15 (StatevectorDistributed).
        bytes[0] = (bytes[0] & !0x20) | 0x40;
        if bytes.len() > 1 {
            bytes[1] = 15;
        }
        let mut case = simulate_case_from_bytes(&bytes);
        case.kind = BackendKind::StatevectorDistributed {
            context: DistributedContext::serial(),
        };
        if let Err(panic) = catch_unwind(AssertUnwindSafe(|| check_simulate_case(&case)))
            && !inside_known_defect(&case)
        {
            resume_unwind(panic);
        }
    }
}
