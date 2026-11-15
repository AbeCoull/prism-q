//! Queries past a dense probability cap pinned through `PRISM_MAX_PROB_QUBITS`. The cap
//! is cached per process, so every module here shares this binary and its one override.

#[path = "../common/mod.rs"]
mod common;

#[cfg(feature = "distributed")]
mod distributed;
mod marginals;
mod tensornetwork_sample;

const PROB_CAP: usize = 4;

fn small_prob_cap() {
    common::caps::set_once(&[("PRISM_MAX_PROB_QUBITS", "4")]);
}
