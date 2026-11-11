//! Shared helpers for the bench targets.
//!
//! Each bench file declares `mod common;` and imports the helpers it
//! needs. Group timing lives here so `--features bench-fast` shortens
//! every target the same way.

#![allow(dead_code)]

pub mod fixtures;
pub mod qec;

use num_complex::Complex64;
use prism_q::circuit::Circuit;
use prism_q::sim;
use prism_q::{BackendKind, PauliObservable, PauliTerm};
use std::time::Duration;

pub const SEED: u64 = 0xDEAD_BEEF;

pub fn run_with(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> prism_q::Result<prism_q::RunOutcome> {
    sim::simulate(circuit).backend(kind).seed(seed).run()
}

pub fn run_shots_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> prism_q::Result<prism_q::ShotsResult> {
    sim::simulate(circuit)
        .backend(kind)
        .seed(seed)
        .shots(num_shots)
}

pub fn is_fast() -> bool {
    cfg!(feature = "bench-fast")
}

/// Criterion instance shared by every bench target.
///
/// Report rendering dominates wall time: on the reference host a five-row group
/// took 335s with plots and 47s without, so roughly 58s per row goes to HTML
/// against 9s of measurement. Regression gating reads stdout and
/// `target/criterion/**/estimates.json`, neither of which the plots feed, so
/// they are off unless `PRISM_BENCH_PLOTS` is set. The confidence-interval bootstrap
/// draws 10,000 resamples rather than 100,000; point estimates do not depend on it.
pub fn criterion_config() -> criterion::Criterion {
    let criterion = criterion::Criterion::default().nresamples(10_000);
    if std::env::var_os("PRISM_BENCH_PLOTS").is_some() {
        criterion
    } else {
        criterion.without_plots()
    }
}

/// Samples per row outside the `bench-fast` tier.
///
/// Criterion splits `measurement_time` across the sample count rather than
/// multiplying by it, so the count is free while one iteration still fits in
/// `measurement_time / samples`. Past that the count sets the cost outright, and
/// the corpus now holds rows well past it: `density_matrix/unitary_layers/12`
/// runs 3.87s per iteration, where 100 samples cost 39 minutes across the six
/// passes of an adjacent A/B against 4 minutes at 10.
///
/// 30 is the compromise. Standard error falls as `1/sqrt(n)`, so dropping from
/// 100 widens it by about 1.8x while cutting slow-row wall time by 3.3x, and a
/// 2026-08-05 A/B at 30 held every control row on the stabilizer and factored
/// groups under 3%.
///
/// `PRISM_BENCH_SAMPLES` raises it for a row that needs a tighter interval, or
/// lowers it for triage. Criterion rejects fewer than 10.
pub fn sample_size() -> usize {
    sample_size_or(30)
}

/// `PRISM_BENCH_SAMPLES` when set, else `default_samples`, floored at Criterion's 10.
fn sample_size_or(default_samples: usize) -> usize {
    std::env::var("PRISM_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default_samples)
        .max(10)
}

pub fn configure_group(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    configure_group_at(group, 30);
}

/// [`configure_group`] with a group's own sample count outside `bench-fast`.
pub fn configure_group_at(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    default_samples: usize,
) {
    if is_fast() {
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
    } else {
        group.sample_size(sample_size_or(default_samples));
    }
}

/// [`configure_group_at`] plus warm-up and measurement times outside `bench-fast`.
pub fn configure_group_timed(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    default_samples: usize,
    warm_up: Duration,
    measurement: Duration,
) {
    configure_group_at(group, default_samples);
    if !is_fast() {
        group.warm_up_time(warm_up);
        group.measurement_time(measurement);
    }
}

/// Transverse-field Ising energy: a `ZZ` chain plus an `X` field, two commuting
/// groups.
pub fn ising_hamiltonian(n: usize) -> PauliObservable {
    let chain = (0..n - 1).map(|q| (1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]));
    let field = (0..n).map(|q| (0.5, vec![PauliTerm::x(q)]));
    PauliObservable::from_terms(chain.chain(field).collect::<Vec<_>>()).unwrap()
}

/// Heisenberg couplings at distance one and two plus fields on all three axes:
/// `9n - 9` terms in three commuting groups, 99 at 12 qubits.
pub fn heisenberg_hamiltonian(n: usize) -> PauliObservable {
    let mut terms = Vec::new();
    for d in 1..=2 {
        for q in 0..n - d {
            terms.push((1.0, vec![PauliTerm::x(q), PauliTerm::x(q + d)]));
            terms.push((0.9, vec![PauliTerm::y(q), PauliTerm::y(q + d)]));
            terms.push((0.8, vec![PauliTerm::z(q), PauliTerm::z(q + d)]));
        }
    }
    for q in 0..n {
        terms.push((0.3, vec![PauliTerm::x(q)]));
        terms.push((0.2, vec![PauliTerm::y(q)]));
        terms.push((0.1, vec![PauliTerm::z(q)]));
    }
    PauliObservable::from_terms(terms).unwrap()
}

/// Kraus set for amplitude damping at rate `gamma`.
pub fn amplitude_damping_kraus(gamma: f64) -> Vec<[[Complex64; 2]; 2]> {
    let c = |re: f64| Complex64::new(re, 0.0);
    let zero = c(0.0);
    vec![
        [[c(1.0), zero], [zero, c((1.0 - gamma).sqrt())]],
        [[zero, c(gamma.sqrt())], [zero, zero]],
    ]
}

/// Kraus pair for correlated `ZZ` dephasing at rate `p`, in the
/// `2*bit(q0) + bit(q1)` packing `NoiseChannel::Kraus2q` takes. Compiles to a diagonal
/// superoperator.
pub fn correlated_zz_kraus(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let zero = Complex64::new(0.0, 0.0);
    let diag = |scale: f64, signs: [f64; 4]| {
        let mut m = [[zero; 4]; 4];
        for (t, sign) in signs.iter().enumerate() {
            m[t][t] = Complex64::new(scale * sign, 0.0);
        }
        m
    };
    vec![
        diag((1.0 - p).sqrt(), [1.0, 1.0, 1.0, 1.0]),
        diag(p.sqrt(), [1.0, -1.0, -1.0, 1.0]),
    ]
}

/// The correlated `ZZ` pair conjugated by `H` on `q0`: `{sqrt(1-p) I, sqrt(p) X(x)Z}`.
/// Unitary conjugation of the whole set preserves CPTP, and the `X` factor fills the
/// compiled 16x16 superoperator, where the `ZZ` pair compiles to its diagonal.
pub fn h_conjugated_zz_kraus(p: f64) -> Vec<[[Complex64; 4]; 4]> {
    let zero = Complex64::new(0.0, 0.0);
    let mut k0 = [[zero; 4]; 4];
    let mut k1 = [[zero; 4]; 4];
    for t in 0..4 {
        k0[t][t] = Complex64::new((1.0 - p).sqrt(), 0.0);
        let sign = if t & 1 == 0 { 1.0 } else { -1.0 };
        k1[t][t ^ 2] = Complex64::new(p.sqrt() * sign, 0.0);
    }
    vec![k0, k1]
}
