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

pub fn configure_group(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    if is_fast() {
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
    } else {
        group.sample_size(10);
    }
}
