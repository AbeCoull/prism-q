//! Tensor-network contractions past `PRISM_MAX_TN_PEAK_QUBITS` and `PRISM_MAX_TN_SLICES`,
//! pinned low enough that ordinary circuits cross them. Both caps are cached per process,
//! so every module here shares this binary and its one pair of overrides.

#[path = "../common/mod.rs"]
mod common;

mod bounded;
mod slicing;

/// Peak cap of `2^10` elements against a slice budget of 64, so a 14-qubit
/// ansatz slices under the cap and a 12-qubit register behind one 10-qubit
/// multi-controlled gate cannot.
const PEAK_CAP: usize = 10;
const SLICE_BUDGET: usize = 64;

fn small_caps() {
    common::caps::set_once(&[
        ("PRISM_MAX_TN_PEAK_QUBITS", "10"),
        ("PRISM_MAX_TN_SLICES", "64"),
    ]);
}
