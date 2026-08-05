//! Shared helpers for the bench targets.
//!
//! Each bench file declares `mod common;` and imports the helpers it
//! needs. Group timing lives here so `--features bench-fast` shortens
//! every target the same way.

#![allow(dead_code)]

use prism_q::BackendKind;
use prism_q::circuit::Circuit;
use prism_q::sim;
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
/// they are off unless `PRISM_BENCH_PLOTS` is set.
pub fn criterion_config() -> criterion::Criterion {
    let criterion = criterion::Criterion::default();
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
    std::env::var("PRISM_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30)
        .max(10)
}

pub fn configure_group(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    if is_fast() {
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
    } else {
        group.sample_size(sample_size());
    }
}
