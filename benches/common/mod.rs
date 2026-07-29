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
/// multiplying by it, so for every row recorded so far (the slowest is 75ms per
/// iteration) raising this from 10 costs almost nothing: the same five-row group
/// took 47s at 10 samples and 48s at 100, while the standard error of the mean
/// falls by roughly three.
///
/// `PRISM_BENCH_SAMPLES` overrides it for opt-in rows whose single iteration is
/// slow enough that the sample count, not the time budget, sets the cost.
/// Criterion rejects fewer than 10.
pub fn sample_size() -> usize {
    std::env::var("PRISM_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100)
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
