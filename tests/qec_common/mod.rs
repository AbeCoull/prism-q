//! Shared helpers for the QEC test targets.

#![allow(dead_code)]

use prism_q::{QecOptions, QecTStrategy};

pub const SEED: u64 = 0xDEAD_BEEF;

pub const ANALYTICAL_STRATEGIES: [QecTStrategy; 3] =
    [QecTStrategy::Auto, QecTStrategy::Spd, QecTStrategy::Camps];

pub fn qec_options(shots: usize, chunk_size: usize, keep_measurements: bool) -> QecOptions {
    QecOptions {
        shots,
        seed: SEED,
        chunk_size: Some(chunk_size),
        keep_measurements,
    }
}
