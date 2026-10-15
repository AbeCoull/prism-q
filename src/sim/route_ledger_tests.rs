// Route ledger: every route dispatch can produce, paired with a test that drives it
// against an oracle independent of the route. Routes gated on a performance
// crossover that small widths do not reach run under `--cfg prism_lowered_thresholds`
// (the `LoweredHere` rows); the default build ignores those tests rather than
// asserting a route the native constant keeps out of reach.

use super::*;
use crate::circuit::{ClassicalCondition, Parameters, PreparedCircuit};
use crate::gates::Gate;
use crate::sim::noise::{NoiseChannel, NoiseEvent};

const SEED: u64 = 42;
const EPS: f64 = 1e-10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Oracle {
    UnfusedStatevector,
    DensityMatrix,
    Tableau,
    ClosedForm,
    UntruncatedMps,
    /// The route exists to decline the terminal; the test asserts a clean error.
    CleanError,
}

#[derive(Clone, Copy, Debug)]
enum Covered {
    Here(&'static str),
    LoweredHere(&'static str),
    Elsewhere {
        file: &'static str,
        test: &'static str,
    },
}

struct Entry {
    terminal: &'static str,
    route: &'static str,
    oracle: Oracle,
    covered: Covered,
}

const fn here(
    terminal: &'static str,
    route: &'static str,
    oracle: Oracle,
    test: &'static str,
) -> Entry {
    Entry {
        terminal,
        route,
        oracle,
        covered: Covered::Here(test),
    }
}

const fn lowered(
    terminal: &'static str,
    route: &'static str,
    oracle: Oracle,
    test: &'static str,
) -> Entry {
    Entry {
        terminal,
        route,
        oracle,
        covered: Covered::LoweredHere(test),
    }
}

const fn elsewhere(
    terminal: &'static str,
    route: &'static str,
    oracle: Oracle,
    file: &'static str,
    test: &'static str,
) -> Entry {
    Entry {
        terminal,
        route,
        oracle,
        covered: Covered::Elsewhere { file, test },
    }
}

use Oracle::{
    CleanError, ClosedForm, DensityMatrix as DmOracle, Tableau, UnfusedStatevector as Sv,
    UntruncatedMps,
};

#[rustfmt::skip]
const LEDGER: &[Entry] = &[
    here("run", "ProbabilityRoute::Direct", Sv, "run_direct_families_match_the_unfused_statevector"),
    here("run", "ExecutionPlan::Backend", Sv, "run_direct_families_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::Statevector", Sv, "run_direct_families_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::ProductState", Sv, "run_direct_families_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::Stabilizer", Sv, "run_direct_families_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::Factored", Sv, "run_direct_families_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::Sparse", Sv, "run_explicit_backends_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::Mps", Sv, "run_explicit_backends_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::TensorNetwork", Sv, "run_explicit_backends_match_the_unfused_statevector"),
    here("run", "ResolvedBackend::DensityMatrix", Sv, "run_explicit_backends_match_the_unfused_statevector"),
    here("run", "ProbabilityRoute::Decomposed", Sv, "run_decomposed_matches_the_unfused_statevector"),
    here("run", "ResolvedBackend::Decomposed", Sv, "run_decomposed_matches_the_unfused_statevector"),
    lowered("run", "ProbabilityRoute::FactoredStabilizer", Sv, "run_factored_stabilizer_matches_the_unfused_statevector"),
    lowered("run", "ResolvedBackend::FactoredStabilizer", Sv, "run_factored_stabilizer_matches_the_unfused_statevector"),
    here("run", "ProbabilityRoute::TemporalClifford", Sv, "run_temporal_clifford_matches_the_unfused_statevector"),
    here("run", "ExecutionPlan::StabilizerRank", Sv, "run_stabilizer_rank_matches_the_unfused_statevector"),
    here("run", "ResolvedBackend::StabilizerRank", Sv, "run_stabilizer_rank_matches_the_unfused_statevector"),
    here("run", "ExecutionPlan::StochasticPauli", CleanError, "run_declines_the_pauli_engines_cleanly"),
    here("run", "ExecutionPlan::DeterministicPauli", CleanError, "run_declines_the_pauli_engines_cleanly"),
    here("run", "ExecutionPlan::PauliPath", CleanError, "run_declines_the_pauli_engines_cleanly"),
    here("run", "exact_mixture", ClosedForm, "noisy_run_reads_the_exact_mixture"),
    here("run", "initial_state", Sv, "initial_state_terminals_match_the_unfused_statevector"),
    elsewhere("run", "ResolvedBackend::Distributed", Sv,
        "src/backend/distributed_statevector/tests.rs",
        "loopback_simulate_starts_the_distributed_backend_from_a_state"),
    elsewhere("run", "ResolvedBackend::Other", CleanError,
        "tests/result_metadata.rs", "every_terminal_stamps_metadata"),
    elsewhere("run", "auto_over_statevector_cap", ClosedForm,
        "tests/statevector_cap_oversize.rs", "require_exact_accepts_a_wide_basis_preserving_circuit"),
    here("run_batch", "run_batch_split", Sv, "run_batch_split_matches_the_unfused_statevector"),
    here("probabilities_of", "probabilities_of", Sv, "probabilities_of_matches_the_unfused_statevector"),

    here("shots", "ShotSource::Compiled", Sv, "compiled_clifford_shots_follow_the_statevector"),
    here("shots", "ResolvedBackend::CompiledStabilizer", Sv, "compiled_clifford_shots_follow_the_statevector"),
    here("shots", "Engine::CompiledSampler", Sv, "compiled_clifford_shots_follow_the_statevector"),
    here("shots", "ShotSource::CompiledDeferred", ClosedForm, "deferred_clifford_shots_follow_the_closed_form"),
    here("shots", "ShotSource::TerminalStatevector", Sv, "terminal_statevector_shots_follow_the_statevector"),
    here("sample_counts", "ShotSource::Native", Sv, "native_sampler_counts_follow_the_statevector"),
    here("sample_counts", "ShotSource::TerminalProbabilities", Sv, "terminal_probability_counts_follow_the_statevector"),
    here("shots", "ShotSource::StabilizerRank", Sv, "explicit_stabilizer_rank_shots_follow_the_statevector"),
    lowered("shots", "clifford_register", Sv, "auto_clifford_register_shots_follow_the_statevector"),
    lowered("shots", "wide_stabilizer_rank", Sv, "auto_wide_stabilizer_rank_shots_follow_the_statevector"),
    here("shots", "ShotSource::PerShot", ClosedForm, "per_shot_route_follows_the_closed_form"),
    here("shots", "initial_state", Sv, "initial_state_terminals_match_the_unfused_statevector"),
    here("sample_counts", "compiled_counts_multinomial", Sv, "compiled_count_regimes_follow_the_statevector"),
    here("sample_counts", "compiled_counts_rank_space", Sv, "compiled_count_regimes_follow_the_statevector"),
    here("sample_counts", "compiled_counts_chunked", Sv, "compiled_count_regimes_follow_the_statevector"),
    here("sample_counts", "terminal_counts_merged_cdf", Sv, "terminal_count_regimes_follow_the_statevector"),
    here("sample_counts", "terminal_counts_searched_dense", Sv, "terminal_count_regimes_follow_the_statevector"),
    lowered("sample_counts", "terminal_counts_searched_sparse", Sv, "terminal_counts_past_the_dense_bin_cap_follow_the_statevector"),
    here("shots", "Engine::FrameSampler", DmOracle, "noisy_clifford_samplers_follow_the_density_matrix"),
    here("shots", "Engine::NoisyCompiledSampler", DmOracle, "noisy_clifford_samplers_follow_the_density_matrix"),
    lowered("shots", "frame_sampler_wide", DmOracle, "wide_frame_sampler_follows_the_density_matrix"),
    here("shots", "trajectory_pauli_groups", DmOracle, "pauli_group_trajectories_follow_the_density_matrix"),
    here("shots", "trajectory_replayed_groups", ClosedForm, "replayed_group_trajectories_follow_the_closed_form"),
    here("shots", "trajectory_per_shot", DmOracle, "general_noise_trajectories_follow_the_density_matrix"),
    here("shots", "exact_mixture", ClosedForm, "noisy_run_reads_the_exact_mixture"),

    here("marginals", "native_marginals", Sv, "marginal_routes_match_the_unfused_statevector"),
    here("marginals", "probability_marginals", Sv, "marginal_routes_match_the_unfused_statevector"),
    here("marginals", "ResolvedBackend::StochasticPauli", Sv, "pauli_engine_marginals_match_the_unfused_statevector"),
    here("marginals", "ResolvedBackend::DeterministicPauli", Sv, "pauli_engine_marginals_match_the_unfused_statevector"),
    here("marginals", "auto_spd_marginals", Sv, "auto_spd_marginals_match_the_unfused_statevector"),
    lowered("marginals", "tensor_route", Sv, "auto_tensor_route_matches_the_unfused_statevector"),
    here("marginals", "exact_mixture", ClosedForm, "noisy_run_reads_the_exact_mixture"),
    here("marginals", "initial_state", Sv, "initial_state_terminals_match_the_unfused_statevector"),

    here("expectation_values", "clifford_pauli_propagation", Sv, "expectation_routes_match_the_unfused_statevector"),
    here("expectation_values", "statevector_expectations", Sv, "expectation_routes_match_the_unfused_statevector"),
    here("expectation_values", "tensor_zero_state", Sv, "expectation_routes_match_the_unfused_statevector"),
    here("expectation_values", "native_expectations", Sv, "expectation_routes_match_the_unfused_statevector"),
    lowered("expectation_values", "tensor_route", Sv, "auto_tensor_route_matches_the_unfused_statevector"),
    here("expectation_values_reported", "ResolvedBackend::StochasticPauli", Sv, "sampled_expectations_carry_errors_that_cover_the_statevector"),
    here("expectation_values", "ResolvedBackend::PauliPath", DmOracle, "pauli_path_matches_the_density_matrix"),
    here("expectation_values", "exact_mixture", ClosedForm, "noisy_run_reads_the_exact_mixture"),
    here("expectation_values", "initial_state", Sv, "initial_state_terminals_match_the_unfused_statevector"),
    elsewhere("expectation_values", "tensor_route_over_mps", UntruncatedMps,
        "tests/tensor_route.rs", "above_the_cap_a_bond_heavy_chain_takes_the_tensor_path"),

    here("observable_expectation", "grouped_pair_masks", Sv, "grouped_observable_routes_match_the_unfused_statevector"),
    here("observable_expectation", "grouped_dedicated_pass", Sv, "grouped_observable_routes_match_the_unfused_statevector"),
    here("observable_expectation", "term_by_term", Sv, "grouped_observable_routes_match_the_unfused_statevector"),
    here("observable_variance", "observable_variance", Sv, "observable_variance_matches_the_unfused_statevector"),

    here("state_vector", "state_export", Sv, "state_diagnostics_match_the_unfused_statevector"),
    here("reduced_density_matrix", "reduced_density_matrix", Sv, "state_diagnostics_match_the_unfused_statevector"),
    here("reduced_density_matrix", "exact_mixture", ClosedForm, "noisy_run_reads_the_exact_mixture"),
    here("entanglement_entropy", "entanglement_entropy", ClosedForm, "entanglement_entropy_matches_the_closed_form"),
    here("overlap", "overlap_native", Tableau, "overlap_matches_the_unfused_statevector"),
    here("overlap", "overlap_dense_export", Sv, "overlap_matches_the_unfused_statevector"),
    here("expectation_gradient", "adjoint_gradient", Sv, "gradients_match_finite_differences"),
    here("expectation_gradient_shift", "shift_gradient", Sv, "gradients_match_finite_differences"),

    here("PreparedCircuit", "PreparedValues::Held", Sv, "prepared_routes_match_the_unfused_statevector"),
    here("PreparedCircuit", "PreparedValues::Simulate", Sv, "prepared_routes_match_the_unfused_statevector"),
    lowered("PreparedCircuit", "PreparedValues::TensorFirst", Sv, "prepared_tensor_first_matches_the_unfused_statevector"),
    here("PreparedCircuit", "PreparedObservable::Held", Sv, "prepared_routes_match_the_unfused_statevector"),
    here("PreparedCircuit", "PreparedObservable::Grouped", Sv, "prepared_routes_match_the_unfused_statevector"),
    here("PreparedCircuit", "PreparedObservable::Simulate", Sv, "prepared_routes_match_the_unfused_statevector"),
];

/// Routes picked by a predicate or a size test rather than an enum variant. A new one
/// is added here by hand together with its ledger row.
const PREDICATE_ROUTES: &[&str] = &[
    "exact_mixture",
    "initial_state",
    "auto_over_statevector_cap",
    "run_batch_split",
    "probabilities_of",
    "clifford_register",
    "wide_stabilizer_rank",
    "compiled_counts_multinomial",
    "compiled_counts_rank_space",
    "compiled_counts_chunked",
    "terminal_counts_merged_cdf",
    "terminal_counts_searched_dense",
    "terminal_counts_searched_sparse",
    "frame_sampler_wide",
    "trajectory_pauli_groups",
    "trajectory_replayed_groups",
    "trajectory_per_shot",
    "native_marginals",
    "probability_marginals",
    "auto_spd_marginals",
    "tensor_route",
    "tensor_route_over_mps",
    "clifford_pauli_propagation",
    "statevector_expectations",
    "tensor_zero_state",
    "native_expectations",
    "grouped_pair_masks",
    "grouped_dedicated_pass",
    "term_by_term",
    "observable_variance",
    "state_export",
    "reduced_density_matrix",
    "entanglement_entropy",
    "overlap_native",
    "overlap_dense_export",
    "adjoint_gradient",
    "shift_gradient",
];

/// Entry points outside `Simulate` that carry their own route choice.
const EXTRA_TERMINALS: &[&str] = &["run_batch", "PreparedCircuit"];

/// One function and one key list per route enum, from a single list of match arms:
/// the match is exhaustive, so a new variant fails to compile until it has an arm,
/// and the arm's key then needs a ledger row before the coverage test passes.
macro_rules! route_keys {
    ($key_of:ident, $all:ident, $ty:ty, { $($(#[$attr:meta])* $pat:pat => $key:literal,)+ }) => {
        fn $key_of(value: &$ty) -> &'static str {
            match value {
                $($(#[$attr])* $pat => $key,)+
            }
        }

        fn $all() -> Vec<&'static str> {
            let mut keys = Vec::new();
            $($(#[$attr])* keys.push($key);)+
            keys
        }
    };
}

route_keys!(shot_source_key, shot_source_keys, ShotSource, {
    ShotSource::Compiled { deferred: false, .. } => "ShotSource::Compiled",
    ShotSource::Compiled { deferred: true, .. } => "ShotSource::CompiledDeferred",
    ShotSource::TerminalStatevector { .. } => "ShotSource::TerminalStatevector",
    ShotSource::Native { .. } => "ShotSource::Native",
    ShotSource::TerminalProbabilities { .. } => "ShotSource::TerminalProbabilities",
    ShotSource::StabilizerRank => "ShotSource::StabilizerRank",
    ShotSource::PerShot => "ShotSource::PerShot",
});

route_keys!(probability_route_key, probability_route_keys, ProbabilityRoute, {
    ProbabilityRoute::FactoredStabilizer => "ProbabilityRoute::FactoredStabilizer",
    ProbabilityRoute::Decomposed(_) => "ProbabilityRoute::Decomposed",
    ProbabilityRoute::TemporalClifford { .. } => "ProbabilityRoute::TemporalClifford",
    ProbabilityRoute::Direct { .. } => "ProbabilityRoute::Direct",
});

route_keys!(execution_plan_key, execution_plan_keys, ExecutionPlan, {
    ExecutionPlan::Backend(_) => "ExecutionPlan::Backend",
    ExecutionPlan::StabilizerRank => "ExecutionPlan::StabilizerRank",
    ExecutionPlan::StochasticPauli { .. } => "ExecutionPlan::StochasticPauli",
    ExecutionPlan::DeterministicPauli { .. } => "ExecutionPlan::DeterministicPauli",
    ExecutionPlan::PauliPath => "ExecutionPlan::PauliPath",
});

route_keys!(resolved_key, resolved_keys, ResolvedBackend, {
    ResolvedBackend::ProductState => "ResolvedBackend::ProductState",
    ResolvedBackend::Stabilizer => "ResolvedBackend::Stabilizer",
    ResolvedBackend::FactoredStabilizer => "ResolvedBackend::FactoredStabilizer",
    ResolvedBackend::Sparse => "ResolvedBackend::Sparse",
    ResolvedBackend::Mps => "ResolvedBackend::Mps",
    ResolvedBackend::Factored => "ResolvedBackend::Factored",
    ResolvedBackend::TensorNetwork => "ResolvedBackend::TensorNetwork",
    ResolvedBackend::Statevector => "ResolvedBackend::Statevector",
    ResolvedBackend::DensityMatrix => "ResolvedBackend::DensityMatrix",
    ResolvedBackend::Distributed => "ResolvedBackend::Distributed",
    ResolvedBackend::StabilizerRank => "ResolvedBackend::StabilizerRank",
    ResolvedBackend::StochasticPauli => "ResolvedBackend::StochasticPauli",
    ResolvedBackend::DeterministicPauli => "ResolvedBackend::DeterministicPauli",
    ResolvedBackend::PauliPath => "ResolvedBackend::PauliPath",
    ResolvedBackend::CompiledStabilizer => "ResolvedBackend::CompiledStabilizer",
    ResolvedBackend::Decomposed => "ResolvedBackend::Decomposed",
    ResolvedBackend::Other(_) => "ResolvedBackend::Other",
});

route_keys!(engine_key, engine_keys, Engine, {
    Engine::CompiledSampler => "Engine::CompiledSampler",
    Engine::NoisyCompiledSampler => "Engine::NoisyCompiledSampler",
    Engine::FrameSampler => "Engine::FrameSampler",
});

route_keys!(prepared_values_key, prepared_values_keys, PreparedValues, {
    PreparedValues::Held => "PreparedValues::Held",
    PreparedValues::TensorFirst => "PreparedValues::TensorFirst",
    PreparedValues::Simulate => "PreparedValues::Simulate",
});

route_keys!(prepared_observable_key, prepared_observable_keys, PreparedObservable, {
    PreparedObservable::Held => "PreparedObservable::Held",
    PreparedObservable::Grouped => "PreparedObservable::Grouped",
    PreparedObservable::Simulate => "PreparedObservable::Simulate",
});

const THIS_FILE: &str = include_str!("route_ledger_tests.rs");

/// Sources an `Elsewhere` row may name.
const OTHER_SOURCES: &[(&str, &str)] = &[
    (
        "src/backend/distributed_statevector/tests.rs",
        include_str!("../backend/distributed_statevector/tests.rs"),
    ),
    (
        "tests/result_metadata.rs",
        include_str!("../../tests/result_metadata.rs"),
    ),
    (
        "tests/statevector_cap_oversize.rs",
        include_str!("../../tests/statevector_cap_oversize.rs"),
    ),
    (
        "tests/tensor_route.rs",
        include_str!("../../tests/tensor_route.rs"),
    ),
];

/// Public method names of the seeded `Simulate` builder, read from this module's
/// parent source so a new terminal shows up without a ledger edit.
fn simulate_terminals() -> Vec<&'static str> {
    let source = include_str!("mod.rs");
    let start = source
        .find("impl<'c> Simulate<'c, Seeded> {")
        .expect("seeded Simulate impl");
    source[start..]
        .lines()
        .take_while(|line| *line != "}")
        .filter_map(|line| line.strip_prefix("    pub fn "))
        .map(|rest| &rest[..rest.find('(').expect("method signature")])
        .collect()
}

fn sorted_unique(mut keys: Vec<&'static str>) -> Vec<&'static str> {
    keys.sort_unstable();
    keys.dedup();
    keys
}

#[test]
fn route_ledger_covers_every_route() {
    let mut expected = shot_source_keys();
    expected.extend(probability_route_keys());
    expected.extend(execution_plan_keys());
    expected.extend(resolved_keys());
    expected.extend(engine_keys());
    expected.extend(prepared_values_keys());
    expected.extend(prepared_observable_keys());
    expected.extend(PREDICATE_ROUTES);
    let expected = sorted_unique(expected);
    let actual = sorted_unique(LEDGER.iter().map(|entry| entry.route).collect());
    assert_eq!(actual, expected, "route ledger coverage");
}

#[test]
fn route_ledger_covers_every_terminal() {
    let mut expected = simulate_terminals();
    assert!(expected.contains(&"run") && expected.contains(&"overlap"));
    expected.extend(EXTRA_TERMINALS);
    let expected = sorted_unique(expected);
    let actual = sorted_unique(LEDGER.iter().map(|entry| entry.terminal).collect());
    assert_eq!(actual, expected, "route ledger terminals");
}

#[test]
fn route_ledger_names_existing_tests() {
    for entry in LEDGER {
        let (source, test) = match entry.covered {
            Covered::Here(test) | Covered::LoweredHere(test) => (THIS_FILE, test),
            Covered::Elsewhere { file, test } => {
                let source = OTHER_SOURCES
                    .iter()
                    .find(|(path, _)| *path == file)
                    .unwrap_or_else(|| panic!("{file} is not in OTHER_SOURCES"))
                    .1;
                (source, test)
            }
        };
        let at = source.find(&format!("fn {test}()")).unwrap_or_else(|| {
            panic!(
                "{} / {} ({:?} oracle): no test named {test}",
                entry.terminal, entry.route, entry.oracle
            )
        });
        let attributes = &source[source[..at].rfind("#[test]").expect("a #[test] attribute")..at];
        let ignored_by_default = attributes.contains("not(prism_lowered_thresholds)");
        assert_eq!(
            ignored_by_default,
            matches!(entry.covered, Covered::LoweredHere(_)),
            "{test}: a lowered row and only a lowered row is ignored outside the lowered build"
        );
    }
}

// ---------------------------------------------------------------------------
// Oracles and fixtures.
// ---------------------------------------------------------------------------

fn unfused_state(circuit: &Circuit) -> Vec<Complex64> {
    let stripped = circuit.without_measurements();
    let mut backend = StatevectorBackend::new(SEED);
    backend
        .init(stripped.num_qubits, stripped.num_classical_bits)
        .unwrap();
    for instruction in &stripped.instructions {
        backend.apply(instruction).unwrap();
    }
    backend.export_statevector().unwrap()
}

fn unfused_probs(circuit: &Circuit) -> Vec<f64> {
    unfused_state(circuit)
        .iter()
        .map(|a| a.norm_sqr())
        .collect()
}

fn pauli_value(state: &[Complex64], observable: &[PauliTerm]) -> f64 {
    let (mut flip, mut sign, mut num_y) = (0usize, 0usize, 0u32);
    for term in observable {
        let bit = 1usize << term.qubit;
        match term.axis {
            crate::sim::unified_pauli::PauliAxis::X => flip |= bit,
            crate::sim::unified_pauli::PauliAxis::Y => {
                flip |= bit;
                sign |= bit;
                num_y += 1;
            }
            crate::sim::unified_pauli::PauliAxis::Z => sign |= bit,
        }
    }
    let phase = Complex64::i().powu(num_y);
    let mut acc = Complex64::new(0.0, 0.0);
    for (j, amp) in state.iter().enumerate() {
        let parity = if (j & sign).count_ones() % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        acc += state[j ^ flip].conj() * phase * parity * amp;
    }
    acc.re
}

fn assert_close(label: &str, got: &[f64], want: &[f64], eps: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= eps, "{label}[{i}]: {g} vs {w}");
    }
}

fn marginals_of(probs: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; 2 * n];
    for (j, p) in probs.iter().enumerate() {
        for q in 0..n {
            out[2 * q + ((j >> q) & 1)] += p;
        }
    }
    out
}

fn flatten(pairs: &[(f64, f64)]) -> Vec<f64> {
    pairs.iter().flat_map(|&(a, b)| [a, b]).collect()
}

fn count_index(key: &[u64]) -> usize {
    key.first().copied().unwrap_or(0) as usize
}

/// Counts against an exact distribution over the classical register: no outcome
/// of probability zero, and every bin within six standard deviations.
fn assert_counts_follow(label: &str, counts: &HashMap<Vec<u64>, u64>, probs: &[f64], shots: usize) {
    let mut binned = vec![0u64; probs.len()];
    for (key, &count) in counts {
        let index = count_index(key);
        assert!(
            index < probs.len(),
            "{label}: outcome {index} outside the register"
        );
        assert!(
            probs[index] > 1e-12,
            "{label}: drew outcome {index} of probability zero"
        );
        binned[index] += count;
    }
    assert_eq!(
        binned.iter().sum::<u64>(),
        shots as u64,
        "{label}: shot total"
    );
    for (index, (&count, &p)) in binned.iter().zip(probs).enumerate() {
        let mean = shots as f64 * p;
        let bound = 6.0 * (mean * (1.0 - p)).sqrt() + 2.0;
        assert!(
            (count as f64 - mean).abs() <= bound,
            "{label}: outcome {index} drawn {count} times, expected {mean:.1}"
        );
    }
}

fn shot_counts(shots: &ShotsResult) -> HashMap<Vec<u64>, u64> {
    shots.counts()
}

fn measure_all(circuit: &Circuit) -> Circuit {
    let mut measured = circuit.clone();
    measured.num_classical_bits = circuit.num_qubits;
    for q in 0..circuit.num_qubits {
        measured.add_measure(q, q);
    }
    measured
}

/// Two brick layers of rotations and CX on a connected register.
fn dense(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..2 {
        for q in 0..n {
            circuit.add_gate(Gate::Ry(0.31 + 0.17 * (q + layer) as f64), &[q]);
            circuit.add_gate(Gate::Rz(0.23 + 0.11 * q as f64), &[q]);
        }
        for q in (layer % 2..n - 1).step_by(2) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    circuit
}

fn clifford(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    circuit.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    circuit.add_gate(Gate::S, &[1]);
    circuit.add_gate(Gate::H, &[n - 1]);
    circuit.add_gate(Gate::Cz, &[0, n - 1]);
    circuit
}

/// `H T H` on the first `t` qubits, then a CX chain over all `n`.
fn clifford_t(n: usize, t: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for q in 0..t {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    circuit
}

fn single_qubit_rotations(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::Ry(0.4 + 0.2 * q as f64), &[q]);
    }
    circuit
}

fn observables(n: usize) -> Vec<Vec<PauliTerm>> {
    vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1), PauliTerm::z(n - 1)],
        vec![PauliTerm::y(0), PauliTerm::y(2)],
        vec![PauliTerm::z(1), PauliTerm::z(2)],
    ]
}

fn run_probs(circuit: &Circuit, kind: BackendKind) -> (Vec<f64>, RunMetadata) {
    let outcome = simulate(circuit).backend(kind).seed(SEED).run().unwrap();
    (outcome.probabilities.unwrap().to_vec(), outcome.metadata)
}

fn noise_on_every_gate(circuit: &Circuit, channel: NoiseChannel) -> NoiseModel {
    let after_gate = circuit
        .instructions
        .iter()
        .map(|inst| match inst {
            Instruction::Gate { targets, .. } => targets
                .iter()
                .map(|&q| NoiseEvent {
                    channel: channel.clone(),
                    qubits: crate::circuit::smallvec![q],
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    NoiseModel {
        after_gate,
        readout: vec![None; circuit.num_classical_bits],
    }
}

fn dm_probs(circuit: &Circuit, noise: &NoiseModel) -> Vec<f64> {
    simulate(circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(noise)
        .seed(SEED)
        .run()
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec()
}

// ---------------------------------------------------------------------------
// Probability routes.
// ---------------------------------------------------------------------------

#[test]
fn run_direct_families_match_the_unfused_statevector() {
    let cases = [
        (dense(6), ResolvedBackend::Statevector),
        (single_qubit_rotations(5), ResolvedBackend::ProductState),
        (clifford(6), ResolvedBackend::Stabilizer),
    ];
    for (circuit, backend) in cases {
        let route = plan_probability_route(
            &BackendKind::Auto,
            &circuit,
            &ResolvedLimits::process_default(),
        );
        assert_eq!(probability_route_key(&route), "ProbabilityRoute::Direct");
        let (_, partial) = analyze_independence(&circuit);
        assert_eq!(
            execution_plan_key(&resolve(
                &BackendKind::Auto,
                &circuit,
                partial,
                &ResolvedLimits::process_default()
            )),
            "ExecutionPlan::Backend"
        );
        let (probs, metadata) = run_probs(&circuit, BackendKind::Auto);
        assert_eq!(resolved_key(&metadata.backend), resolved_key(&backend));
        assert_close(
            resolved_key(&backend),
            &probs,
            &unfused_probs(&circuit),
            EPS,
        );
    }

    // A 6-qubit block and a 2-qubit block: too lopsided to decompose, so the
    // register runs as one factored state.
    let mut partial = dense(6);
    partial.num_qubits = 8;
    partial.add_gate(Gate::Ry(0.7), &[6]);
    partial.add_gate(Gate::Cx, &[6, 7]);
    partial.add_gate(Gate::T, &[7]);
    let (probs, metadata) = run_probs(&partial, BackendKind::Auto);
    assert_eq!(metadata.backend, ResolvedBackend::Factored);
    assert_close("factored", &probs, &unfused_probs(&partial), EPS);
}

#[test]
fn run_explicit_backends_match_the_unfused_statevector() {
    let circuit = dense(6);
    let want = unfused_probs(&circuit);
    let kinds = [
        (BackendKind::Sparse, ResolvedBackend::Sparse),
        (BackendKind::Mps { max_bond_dim: 64 }, ResolvedBackend::Mps),
        (BackendKind::TensorNetwork, ResolvedBackend::TensorNetwork),
        (
            BackendKind::TensorNetworkBounded { tolerance: 1e-12 },
            ResolvedBackend::TensorNetwork,
        ),
        (BackendKind::DensityMatrix, ResolvedBackend::DensityMatrix),
        (BackendKind::Statevector, ResolvedBackend::Statevector),
    ];
    for (kind, backend) in kinds {
        let label = format!("{kind:?}");
        let (probs, metadata) = run_probs(&circuit, kind);
        assert_eq!(metadata.backend, backend, "{label}");
        assert_close(&label, &probs, &want, 1e-9);
    }
}

#[test]
fn run_decomposed_matches_the_unfused_statevector() {
    let mut circuit = dense(4);
    circuit.num_qubits = 8;
    for q in 4..8 {
        circuit.add_gate(Gate::Ry(0.2 * q as f64), &[q]);
    }
    for q in 4..7 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    circuit.add_gate(Gate::T, &[5]);
    let route = plan_probability_route(
        &BackendKind::Auto,
        &circuit,
        &ResolvedLimits::process_default(),
    );
    assert_eq!(
        probability_route_key(&route),
        "ProbabilityRoute::Decomposed"
    );
    let (probs, metadata) = run_probs(&circuit, BackendKind::Auto);
    assert_eq!(metadata.backend, ResolvedBackend::Decomposed);
    assert_close("decomposed", &probs, &unfused_probs(&circuit), EPS);
}

fn bell_pairs(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
        circuit.add_gate(Gate::S, &[q + 1]);
    }
    circuit
}

#[test]
#[cfg_attr(
    not(prism_lowered_thresholds),
    ignore = "reaches its route only under --cfg prism_lowered_thresholds"
)]
fn run_factored_stabilizer_matches_the_unfused_statevector() {
    let circuit = bell_pairs(8);
    let route = plan_probability_route(
        &BackendKind::Auto,
        &circuit,
        &ResolvedLimits::process_default(),
    );
    assert_eq!(
        probability_route_key(&route),
        "ProbabilityRoute::FactoredStabilizer"
    );
    let (probs, metadata) = run_probs(&circuit, BackendKind::Auto);
    assert_eq!(metadata.backend, ResolvedBackend::FactoredStabilizer);
    assert_close("factored stabilizer", &probs, &unfused_probs(&circuit), EPS);
}

#[test]
fn run_temporal_clifford_matches_the_unfused_statevector() {
    let n = 8;
    let mut circuit = Circuit::new(n, 0);
    circuit.add_gate(Gate::H, &[0]);
    for _ in 0..3 {
        for q in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    for q in 0..n {
        circuit.add_gate(Gate::Rx(0.3 + 0.05 * q as f64), &[q]);
    }
    let route = plan_probability_route(
        &BackendKind::Auto,
        &circuit,
        &ResolvedLimits::process_default(),
    );
    assert_eq!(
        probability_route_key(&route),
        "ProbabilityRoute::TemporalClifford"
    );
    let (probs, _) = run_probs(&circuit, BackendKind::Auto);
    assert_close("temporal clifford", &probs, &unfused_probs(&circuit), EPS);
}

#[test]
fn run_stabilizer_rank_matches_the_unfused_statevector() {
    let circuit = clifford_t(5, 2);
    let (_, partial) = analyze_independence(&circuit);
    assert_eq!(
        execution_plan_key(&resolve(
            &BackendKind::StabilizerRank,
            &circuit,
            partial,
            &ResolvedLimits::process_default()
        )),
        "ExecutionPlan::StabilizerRank"
    );
    let (probs, metadata) = run_probs(&circuit, BackendKind::StabilizerRank);
    assert_eq!(metadata.backend, ResolvedBackend::StabilizerRank);
    assert_close("stabilizer rank", &probs, &unfused_probs(&circuit), EPS);
}

#[test]
fn run_declines_the_pauli_engines_cleanly() {
    let circuit = clifford_t(4, 1);
    let kinds = [
        BackendKind::StochasticPauli { num_samples: 64 },
        BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Threshold {
                epsilon: 0.0,
                max_terms: 0,
            },
        },
        BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        },
    ];
    let mut seen = Vec::new();
    for kind in kinds {
        seen.push(execution_plan_key(&resolve(
            &kind,
            &circuit,
            false,
            &ResolvedLimits::process_default(),
        )));
        let err = simulate(&circuit).backend(kind.clone()).seed(SEED).run();
        assert!(
            matches!(err, Err(PrismError::IncompatibleBackend { .. })),
            "{kind:?}: {err:?}"
        );
    }
    assert_eq!(
        seen,
        [
            "ExecutionPlan::StochasticPauli",
            "ExecutionPlan::DeterministicPauli",
            "ExecutionPlan::PauliPath"
        ]
    );
}

#[test]
fn noisy_run_reads_the_exact_mixture() {
    // X then a bit flip of rate p: P(1) = 1 - p, <Z> = 2p - 1, and the shot
    // record carries the same rate.
    let p = 0.2;
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = noise_on_every_gate(
        &circuit,
        NoiseChannel::Pauli {
            px: p,
            py: 0.0,
            pz: 0.0,
        },
    );
    // Flip rates compose: q0 flips once (after X) and again (after CX); q1
    // inherits the first flip and takes its own.
    let flip0 = 2.0 * p * (1.0 - p);
    let p0_one = 1.0 - flip0;
    let dm = BackendKind::DensityMatrix;

    let run = simulate(&circuit)
        .backend(dm.clone())
        .noise(&noise)
        .seed(SEED)
        .run()
        .unwrap();
    assert_eq!(run.metadata.backend, ResolvedBackend::DensityMatrix);
    let probs = run.probabilities.unwrap().to_vec();
    assert!((marginals_of(&probs, 2)[1] - p0_one).abs() < EPS);

    let marginals = simulate(&circuit)
        .backend(dm.clone())
        .noise(&noise)
        .seed(SEED)
        .marginals()
        .unwrap();
    assert!((marginals.marginals[0].1 - p0_one).abs() < EPS);

    let z0 = simulate(&circuit)
        .backend(dm.clone())
        .noise(&noise)
        .seed(SEED)
        .expectation_values(&[vec![PauliTerm::z(0)]])
        .unwrap();
    assert!((z0[0] - (1.0 - 2.0 * p0_one)).abs() < EPS);

    let rho = simulate(&circuit)
        .backend(dm.clone())
        .noise(&noise)
        .seed(SEED)
        .reduced_density_matrix(&[0])
        .unwrap();
    assert!((rho.data[3].re - p0_one).abs() < EPS);
    assert!(rho.data[1].norm() < EPS);

    let mut measured = circuit.clone();
    measured.add_measure(0, 0);
    measured.add_measure(1, 1);
    let mut measured_noise = noise.clone();
    measured_noise.after_gate.extend([Vec::new(), Vec::new()]);
    let shots = 4000;
    let result = simulate(&measured)
        .backend(dm)
        .noise(&measured_noise)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::DensityMatrix);
    assert_counts_follow("dm shots", &shot_counts(&result), &probs, shots);
}

#[test]
fn initial_state_terminals_match_the_unfused_statevector() {
    let n = 4;
    let start: Vec<Complex64> = (0..1 << n)
        .map(|j| Complex64::new(1.0 + j as f64, 0.5 * j as f64))
        .collect();
    let norm = start.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    let start: Vec<Complex64> = start.iter().map(|a| a / norm).collect();
    let circuit = dense(n);

    let mut oracle = StatevectorBackend::new(SEED);
    oracle.init_from_amplitudes(start.clone(), 0).unwrap();
    for instruction in &circuit.instructions {
        oracle.apply(instruction).unwrap();
    }
    let state = oracle.export_statevector().unwrap();
    let probs: Vec<f64> = state.iter().map(|a| a.norm_sqr()).collect();

    let run = simulate(&circuit)
        .initial_state(&start)
        .seed(SEED)
        .run()
        .unwrap();
    assert_close(
        "initial run",
        &run.probabilities.unwrap().to_vec(),
        &probs,
        EPS,
    );

    let marginals = simulate(&circuit)
        .initial_state(&start)
        .seed(SEED)
        .marginals()
        .unwrap();
    assert_close(
        "initial marginals",
        &flatten(&marginals.marginals),
        &marginals_of(&probs, n),
        EPS,
    );

    let obs = observables(n);
    let values = simulate(&circuit)
        .initial_state(&start)
        .seed(SEED)
        .expectation_values(&obs)
        .unwrap();
    let want: Vec<f64> = obs.iter().map(|o| pauli_value(&state, o)).collect();
    assert_close("initial expectations", &values, &want, EPS);

    let measured = measure_all(&circuit);
    let shots = 4000;
    let result = simulate(&measured)
        .initial_state(&start)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_counts_follow("initial shots", &shot_counts(&result), &probs, shots);
}

#[test]
fn run_batch_split_matches_the_unfused_statevector() {
    let circuits: Vec<Circuit> = (3..7).map(dense).collect();
    #[cfg(feature = "parallel")]
    assert!(runs_split_across_workers(&BackendKind::Auto, 6));
    let outcomes = run_batch(&circuits, BackendKind::Auto, SEED).unwrap();
    for (circuit, outcome) in circuits.iter().zip(outcomes) {
        assert_close(
            "batch",
            &outcome.probabilities.unwrap().to_vec(),
            &unfused_probs(circuit),
            EPS,
        );
    }
}

#[test]
fn probabilities_of_matches_the_unfused_statevector() {
    let circuit = dense(5);
    let probs = unfused_probs(&circuit);
    let got = simulate(&circuit)
        .seed(SEED)
        .probabilities_of(&[3, 1])
        .unwrap();
    let mut want = vec![0.0; 4];
    for (j, p) in probs.iter().enumerate() {
        want[((j >> 3) & 1) | (((j >> 1) & 1) << 1)] += p;
    }
    assert_close("probabilities_of", &got, &want, EPS);
}

// ---------------------------------------------------------------------------
// Shot routes.
// ---------------------------------------------------------------------------

fn source_key(kind: &BackendKind, circuit: &Circuit, shots: usize) -> &'static str {
    shot_source_key(
        &prepare_shot_source(
            kind,
            circuit,
            shots,
            SEED,
            &ResolvedLimits::process_default(),
        )
        .unwrap(),
    )
}

#[test]
fn compiled_clifford_shots_follow_the_statevector() {
    let circuit = measure_all(&clifford(6));
    assert_eq!(
        source_key(&BackendKind::Auto, &circuit, 100),
        "ShotSource::Compiled"
    );
    let shots = 4000;
    let result = simulate(&circuit).seed(SEED).shots(shots).unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::CompiledStabilizer);
    assert_eq!(
        result.metadata.engine.as_ref().map(engine_key),
        Some("Engine::CompiledSampler")
    );
    assert_counts_follow(
        "compiled",
        &shot_counts(&result),
        &unfused_probs(&circuit),
        shots,
    );
}

#[test]
fn deferred_clifford_shots_follow_the_closed_form() {
    // c0 = H|0> measured, reset, then c1 = (H|0> measured) xor c0 via CX from a
    // fresh qubit: four outcomes, uniform.
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_reset(0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(1, 1);
    assert_eq!(
        source_key(&BackendKind::Auto, &circuit, 100),
        "ShotSource::CompiledDeferred"
    );
    let shots = 4000;
    let result = simulate(&circuit).seed(SEED).shots(shots).unwrap();
    assert_counts_follow("deferred", &shot_counts(&result), &[0.25; 4], shots);
}

#[test]
fn terminal_statevector_shots_follow_the_statevector() {
    let circuit = measure_all(&dense(6));
    assert_eq!(
        source_key(&BackendKind::Auto, &circuit, 100),
        "ShotSource::TerminalStatevector"
    );
    let shots = 4000;
    let result = simulate(&circuit).seed(SEED).shots(shots).unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::Statevector);
    assert_counts_follow(
        "terminal statevector",
        &shot_counts(&result),
        &unfused_probs(&circuit),
        shots,
    );
}

#[test]
fn native_sampler_counts_follow_the_statevector() {
    let circuit = measure_all(&dense(6));
    let kind = BackendKind::Mps { max_bond_dim: 64 };
    assert_eq!(source_key(&kind, &circuit, 100), "ShotSource::Native");
    let shots = 4000;
    let counts = simulate(&circuit)
        .backend(kind)
        .seed(SEED)
        .sample_counts(shots)
        .unwrap();
    assert_eq!(counts.metadata.backend, ResolvedBackend::Mps);
    assert_counts_follow("native", &counts.counts, &unfused_probs(&circuit), shots);
}

fn decomposable(n: usize) -> Circuit {
    let half = n / 2;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::Ry(0.35 + 0.1 * q as f64), &[q]);
    }
    for q in (0..half - 1).chain(half..n - 1) {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    circuit.add_gate(Gate::T, &[1]);
    circuit
}

#[test]
fn terminal_probability_counts_follow_the_statevector() {
    let circuit = measure_all(&decomposable(8));
    assert_eq!(
        source_key(&BackendKind::Auto, &circuit, 100),
        "ShotSource::TerminalProbabilities"
    );
    let shots = 4000;
    let counts = simulate(&circuit).seed(SEED).sample_counts(shots).unwrap();
    assert_eq!(counts.metadata.backend, ResolvedBackend::Decomposed);
    assert_counts_follow(
        "terminal probabilities",
        &counts.counts,
        &unfused_probs(&circuit),
        shots,
    );
}

#[test]
fn explicit_stabilizer_rank_shots_follow_the_statevector() {
    let circuit = measure_all(&clifford_t(5, 2));
    assert_eq!(
        source_key(&BackendKind::StabilizerRank, &circuit, 100),
        "ShotSource::StabilizerRank"
    );
    let shots = 4000;
    let result = simulate(&circuit)
        .backend(BackendKind::StabilizerRank)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::StabilizerRank);
    assert_counts_follow(
        "explicit stabilizer rank",
        &shot_counts(&result),
        &unfused_probs(&circuit),
        shots,
    );
}

#[test]
#[cfg_attr(
    not(prism_lowered_thresholds),
    ignore = "reaches its route only under --cfg prism_lowered_thresholds"
)]
fn auto_clifford_register_shots_follow_the_statevector() {
    let circuit = measure_all(&clifford_t(5, 2));
    assert!(auto_prefers_clifford_register(&circuit));
    assert_eq!(
        source_key(&BackendKind::Auto, &circuit, 100),
        "ShotSource::StabilizerRank"
    );
    let shots = 4000;
    let result = simulate(&circuit).seed(SEED).shots(shots).unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::StabilizerRank);
    assert_counts_follow(
        "clifford register",
        &shot_counts(&result),
        &unfused_probs(&circuit),
        shots,
    );
}

#[test]
#[cfg_attr(
    not(prism_lowered_thresholds),
    ignore = "reaches its route only under --cfg prism_lowered_thresholds"
)]
fn auto_wide_stabilizer_rank_shots_follow_the_statevector() {
    // Three T gates exceed the lowered register cap, so the register declines. A
    // detached Bell pair leaves the register partly independent, which keeps it
    // off the terminal statevector, and the branch expansion takes the wide route.
    let mut circuit = clifford_t(10, 3);
    circuit.num_qubits = 12;
    circuit.add_gate(Gate::H, &[10]);
    circuit.add_gate(Gate::Cx, &[10, 11]);
    let circuit = measure_all(&circuit);
    assert!(!auto_prefers_clifford_register(&circuit));
    assert!(auto_samples_wide_by_stabilizer_rank(&circuit));
    let shots = 4000;
    let result = simulate(&circuit).seed(SEED).shots(shots).unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::StabilizerRank);
    assert_counts_follow(
        "wide stabilizer rank",
        &shot_counts(&result),
        &unfused_probs(&circuit),
        shots,
    );
}

#[test]
fn per_shot_route_follows_the_closed_form() {
    // c0 reads Ry(theta)|0>, then a conditional X copies it onto q1, so c1 == c0
    // with P(1) = sin^2(theta / 2).
    let theta: f64 = 1.1;
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::Ry(theta), &[0]);
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: crate::circuit::smallvec![1],
    });
    circuit.add_measure(1, 1);
    assert_eq!(
        source_key(&BackendKind::Auto, &circuit, 100),
        "ShotSource::PerShot"
    );
    let p1 = (theta / 2.0).sin().powi(2);
    let shots = 4000;
    let result = simulate(&circuit).seed(SEED).shots(shots).unwrap();
    assert_counts_follow(
        "per shot",
        &shot_counts(&result),
        &[1.0 - p1, 0.0, 0.0, p1],
        shots,
    );
}

#[test]
fn compiled_count_regimes_follow_the_statevector() {
    // Rank 3, eight equally likely parity outcomes: the multinomial path from
    // 64 shots, the rank-space path from 32, and the chunked sampler below.
    let mut base = Circuit::new(6, 0);
    for q in 0..3 {
        base.add_gate(Gate::H, &[q]);
    }
    for (a, b) in [(0, 3), (1, 4), (2, 5), (3, 4), (4, 5)] {
        base.add_gate(Gate::Cx, &[a, b]);
    }
    let circuit = measure_all(&base);
    let probs = unfused_probs(&circuit);
    for shots in [20_000, 40, 20] {
        let counts = simulate(&circuit).seed(SEED).sample_counts(shots).unwrap();
        assert_eq!(counts.metadata.backend, ResolvedBackend::CompiledStabilizer);
        assert_counts_follow(
            &format!("compiled counts {shots}"),
            &counts.counts,
            &probs,
            shots,
        );
    }
}

#[test]
fn terminal_count_regimes_follow_the_statevector() {
    // Shots at or under the outcome count take the merged CDF; more shots than
    // outcomes keep the searched CDF with dense bins.
    let circuit = measure_all(&dense(4));
    let probs = unfused_probs(&circuit);
    for shots in [16, 8, 4000] {
        let counts = simulate(&circuit).seed(SEED).sample_counts(shots).unwrap();
        assert_eq!(counts.metadata.backend, ResolvedBackend::Statevector);
        assert_counts_follow(
            &format!("terminal counts {shots}"),
            &counts.counts,
            &probs,
            shots,
        );
    }
}

#[test]
#[cfg_attr(
    not(prism_lowered_thresholds),
    ignore = "reaches its route only under --cfg prism_lowered_thresholds"
)]
fn terminal_counts_past_the_dense_bin_cap_follow_the_statevector() {
    let circuit = measure_all(&dense(6));
    let shots = 8000;
    let counts = simulate(&circuit).seed(SEED).sample_counts(shots).unwrap();
    assert_counts_follow(
        "sparse bins",
        &counts.counts,
        &unfused_probs(&circuit),
        shots,
    );
}

/// Measured Clifford layers of `n + 1` gates each, so the depth ratio the noisy
/// sampler choice reads is `layers * (n + 1) / n`.
fn clifford_layers(n: usize, layers: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..layers {
        for q in 0..n {
            if (q + layer) % 2 == 0 {
                circuit.add_gate(Gate::H, &[q]);
            } else {
                circuit.add_gate(Gate::S, &[q]);
            }
        }
        let a = layer % (n - 1);
        circuit.add_gate(Gate::Cx, &[a, a + 1]);
    }
    measure_all(&circuit)
}

#[test]
fn noisy_clifford_samplers_follow_the_density_matrix() {
    let shots = 4000;
    for (layers, engine) in [
        (1, "Engine::FrameSampler"),
        (8, "Engine::NoisyCompiledSampler"),
    ] {
        let circuit = clifford_layers(4, layers);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.05);
        let result = simulate(&circuit)
            .noise(&noise)
            .seed(SEED)
            .shots(shots)
            .unwrap();
        assert_eq!(result.metadata.backend, ResolvedBackend::CompiledStabilizer);
        assert_eq!(
            result.metadata.engine.as_ref().map(engine_key),
            Some(engine)
        );
        assert_counts_follow(
            engine,
            &shot_counts(&result),
            &dm_probs(&circuit, &noise),
            shots,
        );
    }
}

#[test]
#[cfg_attr(
    not(prism_lowered_thresholds),
    ignore = "reaches its route only under --cfg prism_lowered_thresholds"
)]
fn wide_frame_sampler_follows_the_density_matrix() {
    // Four gates per qubit sits between the narrow and the wide depth cutoffs, so
    // only the wide-register rule sends it to the frame sampler.
    let circuit = clifford_layers(8, 4);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.05);
    let shots = 4000;
    let result = simulate(&circuit)
        .noise(&noise)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(
        result.metadata.engine.as_ref().map(engine_key),
        Some("Engine::FrameSampler")
    );
    assert_counts_follow(
        "wide frame",
        &shot_counts(&result),
        &dm_probs(&circuit, &noise),
        shots,
    );
}

#[test]
fn pauli_group_trajectories_follow_the_density_matrix() {
    let circuit = measure_all(&dense(4));
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
    let shots = 4000;
    assert!(trajectory::PauliGroups::sample(&circuit, &noise, shots, SEED).is_some());
    let result = simulate(&circuit)
        .noise(&noise)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::Statevector);
    assert_counts_follow(
        "pauli groups",
        &shot_counts(&result),
        &dm_probs(&circuit, &noise),
        shots,
    );
}

#[test]
fn replayed_group_trajectories_follow_the_closed_form() {
    // A non-Clifford gate keeps the circuit off the Clifford samplers, and the
    // mid-circuit measurement sends it to replayed groups. c0 = 1 unless the
    // first X flips back (rate p); c1 = not c0 unless the second X flips.
    let p = 0.1;
    let mut circuit = Circuit::new(1, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_measure(0, 1);
    let mut noise = NoiseModel {
        after_gate: vec![Vec::new(); circuit.instructions.len()],
        readout: vec![None; 2],
    };
    noise.after_gate[0].push(NoiseEvent::pauli(0, p, 0.0, 0.0));
    noise.after_gate[3].push(NoiseEvent::pauli(0, p, 0.0, 0.0));
    let shots = 4000;
    assert!(trajectory::PauliGroups::sample(&circuit, &noise, shots, SEED).is_none());
    assert!(trajectory::PauliGroups::sample_replayed(&circuit, &noise, shots, SEED).is_some());
    let result = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    // Index is c0 + 2 c1.
    let want = [p * p, (1.0 - p) * (1.0 - p), p * (1.0 - p), (1.0 - p) * p];
    assert_counts_follow("replayed groups", &shot_counts(&result), &want, shots);
}

#[test]
fn general_noise_trajectories_follow_the_density_matrix() {
    let circuit = measure_all(&dense(4));
    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.05);
    let shots = 4000;
    assert!(trajectory::PauliGroups::sample(&circuit, &noise, shots, SEED).is_none());
    let result = simulate(&circuit)
        .noise(&noise)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::Statevector);
    assert_counts_follow(
        "trajectories",
        &shot_counts(&result),
        &dm_probs(&circuit, &noise),
        shots,
    );
}

// ---------------------------------------------------------------------------
// Marginal and expectation routes.
// ---------------------------------------------------------------------------

#[test]
fn marginal_routes_match_the_unfused_statevector() {
    // Explicit kinds, so neither the Pauli nor the tensor route of `Auto` answers.
    // The stabilizer answers natively; a decomposed register has no single backend
    // to ask and reads the merged distribution.
    let cases = [
        (
            clifford(6),
            BackendKind::Stabilizer,
            ResolvedBackend::Stabilizer,
        ),
        (
            decomposable(8),
            BackendKind::Factored,
            ResolvedBackend::Decomposed,
        ),
    ];
    for (circuit, kind, backend) in cases {
        let result = simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .marginals()
            .unwrap();
        assert_eq!(result.metadata.backend, backend);
        assert_close(
            resolved_key(&backend),
            &flatten(&result.marginals),
            &marginals_of(&unfused_probs(&circuit), circuit.num_qubits),
            EPS,
        );
    }
}

#[test]
fn pauli_engine_marginals_match_the_unfused_statevector() {
    let circuit = clifford_t(5, 2);
    let want = marginals_of(&unfused_probs(&circuit), 5);
    let spd = simulate(&circuit)
        .backend(BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Threshold {
                epsilon: 0.0,
                max_terms: 0,
            },
        })
        .seed(SEED)
        .marginals()
        .unwrap();
    assert_eq!(spd.metadata.backend, ResolvedBackend::DeterministicPauli);
    assert_close("spd", &flatten(&spd.marginals), &want, EPS);

    let samples = 20_000;
    let spp = simulate(&circuit)
        .backend(BackendKind::StochasticPauli {
            num_samples: samples,
        })
        .seed(SEED)
        .marginals()
        .unwrap();
    assert_eq!(spp.metadata.backend, ResolvedBackend::StochasticPauli);
    assert_close("spp", &flatten(&spp.marginals), &want, 0.05);
}

#[test]
fn auto_spd_marginals_match_the_unfused_statevector() {
    // Twelve qubits is the native floor, and the dense run the Pauli sum is priced
    // against grows as 2^n, so narrower registers rarely fit its budget anyway.
    let circuit = clifford_t(12, 2);
    let result = simulate(&circuit).seed(SEED).marginals().unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::DeterministicPauli);
    assert_close(
        "auto spd",
        &flatten(&result.marginals),
        &marginals_of(&unfused_probs(&circuit), 12),
        EPS,
    );
}

#[test]
#[cfg_attr(
    not(prism_lowered_thresholds),
    ignore = "reaches its route only under --cfg prism_lowered_thresholds"
)]
fn auto_tensor_route_matches_the_unfused_statevector() {
    let n = 6;
    let circuit = dense(n);
    let state = unfused_state(&circuit);
    let obs = observables(n);
    assert!(
        decide_tensor_expectations(&circuit, &obs, &ResolvedLimits::process_default()).is_some()
    );

    let result = simulate(&circuit)
        .seed(SEED)
        .expectation_values_reported(&obs)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::TensorNetwork);
    let want: Vec<f64> = obs.iter().map(|o| pauli_value(&state, o)).collect();
    assert_close("tensor expectations", &result.values, &want, EPS);

    let marginals = simulate(&circuit).seed(SEED).marginals().unwrap();
    assert_eq!(marginals.metadata.backend, ResolvedBackend::TensorNetwork);
    assert_close(
        "tensor marginals",
        &flatten(&marginals.marginals),
        &marginals_of(&unfused_probs(&circuit), n),
        EPS,
    );
}

#[test]
fn expectation_routes_match_the_unfused_statevector() {
    let n = 6;
    let cases = [
        (
            clifford(n),
            BackendKind::Auto,
            ResolvedBackend::DeterministicPauli,
        ),
        (
            dense(n),
            BackendKind::Statevector,
            ResolvedBackend::Statevector,
        ),
        (
            dense(n),
            BackendKind::TensorNetwork,
            ResolvedBackend::TensorNetwork,
        ),
        (
            dense(n),
            BackendKind::Mps { max_bond_dim: 64 },
            ResolvedBackend::Mps,
        ),
        (dense(n), BackendKind::Sparse, ResolvedBackend::Sparse),
        (
            decomposable(8),
            BackendKind::Factored,
            ResolvedBackend::Factored,
        ),
        (
            single_qubit_rotations(n),
            BackendKind::ProductState,
            ResolvedBackend::ProductState,
        ),
        (
            clifford(n),
            BackendKind::Stabilizer,
            ResolvedBackend::DeterministicPauli,
        ),
    ];
    for (circuit, kind, backend) in cases {
        let label = format!("{kind:?}");
        let obs = observables(n);
        let state = unfused_state(&circuit);
        let result = simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .expectation_values_reported(&obs)
            .unwrap();
        assert_eq!(result.metadata.backend, backend, "{label}");
        let want: Vec<f64> = obs.iter().map(|o| pauli_value(&state, o)).collect();
        assert_close(&label, &result.values, &want, 1e-9);
    }
}

#[test]
fn sampled_expectations_carry_errors_that_cover_the_statevector() {
    let circuit = clifford_t(5, 2);
    let state = unfused_state(&circuit);
    let obs = observables(5);
    let result = simulate(&circuit)
        .backend(BackendKind::StochasticPauli {
            num_samples: 20_000,
        })
        .seed(SEED)
        .expectation_values_reported(&obs)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::StochasticPauli);
    let errors = result.std_errors.unwrap();
    for ((got, err), o) in result.values.iter().zip(&errors).zip(&obs) {
        let want = pauli_value(&state, o);
        assert!(
            (got - want).abs() <= 6.0 * err + 1e-9,
            "{got} vs {want} +- {err}"
        );
    }
}

#[test]
fn pauli_path_matches_the_density_matrix() {
    let circuit = dense(4);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.03);
    let obs = observables(4);
    let kind = BackendKind::PauliPath {
        epsilon: 0.0,
        max_terms: 0,
    };
    assert_eq!(
        execution_plan_key(&resolve(
            &kind,
            &circuit,
            false,
            &ResolvedLimits::process_default()
        )),
        "ExecutionPlan::PauliPath"
    );
    let path = simulate(&circuit)
        .backend(kind)
        .noise(&noise)
        .seed(SEED)
        .expectation_values_reported(&obs)
        .unwrap();
    assert_eq!(path.metadata.backend, ResolvedBackend::PauliPath);
    let dm = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(SEED)
        .expectation_values(&obs)
        .unwrap();
    assert_close("pauli path", &path.values, &dm, 1e-9);
}

fn many_term_observable(n: usize) -> PauliObservable {
    // Eight Z-only terms share one commuting group: 28 pairwise products, past the
    // pair-mask budget, so the group takes the dedicated pass. The X terms form
    // small groups that stay on the pair masks.
    let mut terms: Vec<(f64, Vec<PauliTerm>)> = (0..n)
        .map(|q| (0.3 + 0.1 * q as f64, vec![PauliTerm::z(q)]))
        .collect();
    terms.push((0.7, vec![PauliTerm::z(0), PauliTerm::z(1)]));
    terms.push((-0.4, vec![PauliTerm::z(2), PauliTerm::z(5)]));
    terms.push((0.5, vec![PauliTerm::x(0), PauliTerm::x(1)]));
    terms.push((0.2, vec![PauliTerm::y(3)]));
    terms.push((1.5, Vec::new()));
    PauliObservable::from_terms(terms).unwrap()
}

fn observable_value(state: &[Complex64], observable: &PauliObservable) -> f64 {
    observable
        .terms()
        .iter()
        .map(|(c, factors)| c * pauli_value(state, factors))
        .sum()
}

#[test]
fn grouped_observable_routes_match_the_unfused_statevector() {
    let n = 6;
    let observable = many_term_observable(n);
    for (circuit, kind, grouped) in [
        (dense(n), BackendKind::Auto, true),
        (dense(n), BackendKind::Statevector, true),
        (clifford(n), BackendKind::Auto, false),
        (dense(n), BackendKind::Mps { max_bond_dim: 64 }, false),
    ] {
        let label = format!("{kind:?}");
        let state = unfused_state(&circuit);
        let result = simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .observable_expectation(&observable)
            .unwrap();
        assert_eq!(result.variance.is_some(), grouped, "{label}");
        assert!(
            (result.mean - observable_value(&state, &observable)).abs() < 1e-9,
            "{label}"
        );
    }
}

#[test]
fn observable_variance_matches_the_unfused_statevector() {
    let n = 6;
    let circuit = dense(n);
    let state = unfused_state(&circuit);
    let observable = many_term_observable(n);
    let mean = observable_value(&state, &observable);
    let square = observable.square();
    let second = observable_value(&state, &square);
    let result = simulate(&circuit)
        .seed(SEED)
        .observable_variance(&observable)
        .unwrap();
    assert!((result.mean - mean).abs() < 1e-9);
    assert!((result.variance - (second - mean * mean)).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// State diagnostics and gradients.
// ---------------------------------------------------------------------------

fn reduced_density(state: &[Complex64], qubits: &[usize]) -> Vec<Complex64> {
    let k = qubits.len();
    let side = 1usize << k;
    let mask: usize = qubits.iter().map(|&q| 1 << q).sum();
    let deposit = |t: usize| -> usize {
        (0..k)
            .filter(|&i| (t >> i) & 1 == 1)
            .map(|i| 1 << qubits[i])
            .sum()
    };
    let extract = |j: usize| -> usize { (0..k).map(|i| ((j >> qubits[i]) & 1) << i).sum() };
    let mut rho = vec![Complex64::new(0.0, 0.0); side * side];
    for (a, amp) in state.iter().enumerate() {
        let rest = a & !mask;
        for t in 0..side {
            let b = rest | deposit(t);
            rho[extract(a) * side + t] += amp * state[b].conj();
        }
    }
    rho
}

#[test]
fn state_diagnostics_match_the_unfused_statevector() {
    let n = 5;
    for (circuit, kind) in [
        (dense(n), BackendKind::Auto),
        (dense(n), BackendKind::Mps { max_bond_dim: 64 }),
        (clifford(n), BackendKind::Stabilizer),
        (dense(n), BackendKind::Sparse),
    ] {
        let label = format!("{kind:?}");
        let state = unfused_state(&circuit);
        let got = simulate(&circuit)
            .backend(kind.clone())
            .seed(SEED)
            .state_vector()
            .unwrap();
        let fidelity = got
            .iter()
            .zip(&state)
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>()
            .norm_sqr();
        assert!(
            (fidelity - 1.0).abs() < 1e-9,
            "{label}: fidelity {fidelity}"
        );

        let qubits = [3, 0];
        let rho = simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .reduced_density_matrix(&qubits)
            .unwrap();
        let want = reduced_density(&state, &qubits);
        for (g, w) in rho.data.iter().zip(&want) {
            assert!((g - w).norm() < 1e-9, "{label}: {g} vs {w}");
        }
    }
}

#[test]
fn entanglement_entropy_matches_the_closed_form() {
    // Ry(theta) then CX: Schmidt weights cos^2(theta / 2) and sin^2(theta / 2).
    let theta: f64 = 0.9;
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::Ry(theta), &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[2]);
    let (c, s) = ((theta / 2.0).cos().powi(2), (theta / 2.0).sin().powi(2));
    let want = -(c * c.ln() + s * s.ln());
    for kind in [
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::Mps { max_bond_dim: 8 },
    ] {
        let result = simulate(&circuit)
            .backend(kind.clone())
            .seed(SEED)
            .entanglement_entropy(&[0])
            .unwrap();
        assert!((result.entropy - want).abs() < 1e-9, "{kind:?}");
    }
}

#[test]
fn overlap_matches_the_unfused_statevector() {
    let n = 5;
    let (a, b) = (clifford(n), {
        let mut b = clifford(n);
        b.add_gate(Gate::H, &[2]);
        b
    });
    let want = |x: &Circuit, y: &Circuit| {
        unfused_state(x)
            .iter()
            .zip(&unfused_state(y))
            .map(|(p, q)| p.conj() * q)
            .sum::<Complex64>()
            .norm_sqr()
    };
    let native = simulate(&a)
        .backend(BackendKind::Stabilizer)
        .seed(SEED)
        .overlap(simulate(&b).backend(BackendKind::Stabilizer).seed(SEED))
        .unwrap();
    assert!((native.fidelity - want(&a, &b)).abs() < EPS);

    let (c, d) = (dense(n), single_qubit_rotations(n));
    let mixed = simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .overlap(
            simulate(&d)
                .backend(BackendKind::Mps { max_bond_dim: 16 })
                .seed(SEED),
        )
        .unwrap();
    assert!((mixed.fidelity - want(&c, &d)).abs() < EPS);
}

#[test]
fn gradients_match_finite_differences() {
    let n = 4;
    let circuit = dense(n);
    let params = Parameters::all_rotations(&circuit);
    let values = params.values(&circuit).unwrap();
    let hamiltonian = vec![
        (0.8, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (-0.5, vec![PauliTerm::x(2)]),
    ];
    let energy = |values: &[f64]| -> f64 {
        let bound = params.bind(&circuit, values).unwrap();
        let state = unfused_state(&bound);
        hamiltonian
            .iter()
            .map(|(c, factors)| c * pauli_value(&state, factors))
            .sum()
    };
    let h = 1e-5;
    let want: Vec<f64> = (0..values.len())
        .map(|i| {
            let mut up = values.clone();
            let mut down = values.clone();
            up[i] += h;
            down[i] -= h;
            (energy(&up) - energy(&down)) / (2.0 * h)
        })
        .collect();

    let adjoint = simulate(&circuit)
        .seed(SEED)
        .expectation_gradient(&hamiltonian, &params)
        .unwrap();
    assert!((adjoint.value - energy(&values)).abs() < EPS);
    assert_close("adjoint", &adjoint.gradient, &want, 1e-6);

    let shift = simulate(&circuit)
        .backend(BackendKind::Mps { max_bond_dim: 64 })
        .seed(SEED)
        .expectation_gradient_shift(&hamiltonian, &params)
        .unwrap();
    assert_close("shift", &shift.gradient, &want, 1e-6);
}

// ---------------------------------------------------------------------------
// Prepared templates.
// ---------------------------------------------------------------------------

fn prepared_keys(kind: &BackendKind, template: &Circuit) -> (&'static str, &'static str) {
    let route = prepared_route(kind, template, None).expect("a direct route");
    (
        prepared_values_key(&route.values),
        prepared_observable_key(&route.observable),
    )
}

fn assert_prepared_matches(kind: BackendKind, template: Circuit, params: Parameters) {
    let label = format!("{kind:?}");
    let values = params.values(&template).unwrap();
    let state = unfused_state(&params.bind(&template, &values).unwrap());
    let obs = observables(template.num_qubits);
    let observable = many_term_observable(template.num_qubits);
    let mut prepared = PreparedCircuit::with_backend(template, params, kind).unwrap();
    let got = prepared.expectation_values(&values, &obs, SEED).unwrap();
    let want: Vec<f64> = obs.iter().map(|o| pauli_value(&state, o)).collect();
    assert_close(&label, &got, &want, 1e-9);
    let mean = prepared
        .observable_expectation(&values, &observable, SEED)
        .unwrap()
        .mean;
    assert!(
        (mean - observable_value(&state, &observable)).abs() < 1e-9,
        "{label}"
    );
}

#[test]
fn prepared_routes_match_the_unfused_statevector() {
    let n = 6;
    let template = dense(n);
    assert_eq!(
        prepared_keys(&BackendKind::Statevector, &template),
        ("PreparedValues::Held", "PreparedObservable::Grouped")
    );
    assert_prepared_matches(
        BackendKind::Statevector,
        template.clone(),
        Parameters::all_rotations(&template),
    );

    let mps = BackendKind::Mps { max_bond_dim: 64 };
    assert_eq!(
        prepared_keys(&mps, &template),
        ("PreparedValues::Held", "PreparedObservable::Held")
    );
    assert_prepared_matches(mps, template.clone(), Parameters::all_rotations(&template));

    let fixed = clifford(n);
    assert_eq!(
        prepared_keys(&BackendKind::Auto, &fixed),
        ("PreparedValues::Simulate", "PreparedObservable::Simulate")
    );
    assert_prepared_matches(BackendKind::Auto, fixed, Parameters::new(0));
}

#[test]
#[cfg_attr(
    not(prism_lowered_thresholds),
    ignore = "reaches its route only under --cfg prism_lowered_thresholds"
)]
fn prepared_tensor_first_matches_the_unfused_statevector() {
    let template = dense(6);
    assert_eq!(
        prepared_keys(&BackendKind::Auto, &template).0,
        "PreparedValues::TensorFirst"
    );
    assert_prepared_matches(
        BackendKind::Auto,
        template.clone(),
        Parameters::all_rotations(&template),
    );
}
