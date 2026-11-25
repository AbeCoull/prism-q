//! Noise channels, noise models, and noisy samplers. Pauli-only models on
//! Clifford circuits compile into a [`NoisyCompiledSampler`]; general
//! channels run through the trajectory engine.

pub mod builder;
mod compiled;
mod frame;
#[cfg(feature = "gpu")]
mod gpu_cache;
mod guarded;
mod mixture;
mod model;

pub use builder::{DriftDistribution, GateFilter, NoiseBuilder};
pub use compiled::*;
pub(in crate::sim) use frame::*;
pub(crate) use guarded::*;
pub use mixture::*;
pub use model::*;

pub(crate) use crate::sim::trajectory::run_shots_noisy_brute_with;

use crate::backend::stabilizer::StabilizerBackend;
use crate::circuit::Circuit;
use crate::error::Result;
use crate::limits::ResolvedLimits;
use crate::sim::ShotsResult;
use crate::sim::compiled::{DeferredCircuit, defer_noisy_circuit};

/// Sample a Pauli-noisy Clifford circuit through the compiled noisy sampler or
/// the frame sampler.
///
/// A circuit with resets, mid-circuit measurements, or guarded Pauli gates is
/// first rewritten to terminal measurements: a reset moves its qubit to a fresh
/// one, a measured qubit used again copies its record onto a fresh one, and a
/// guarded Pauli becomes that Pauli controlled by the measured qubits its
/// condition reads, which holds for a single bit, a parity, or a register
/// comparison reading one measured bit. A comparison reading several measured
/// bits lands its Pauli as record flips on the shots it holds. A circuit the rewrite declines, such as
/// one guarding a gate that is not a Pauli, falls back to per-shot stabilizer
/// replay. Non-Clifford circuits take the same fallback and error on
/// the first non-Clifford gate; the public entry point routes them to the
/// trajectory engine before reaching here.
pub fn run_shots_noisy(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    super::decline_saves(circuit, "NoisyCompiledSampler", "noisy sampling")?;
    noise.validate_for(circuit)?;
    run_validated_shots_noisy(circuit, noise, num_shots, seed)
}

/// [`run_shots_noisy`] for a model already checked by [`NoiseModel::validate_for`].
pub(crate) fn run_validated_shots_noisy(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    noise.ensure_only_pauli_channels()?;
    if !circuit.is_clifford_only() {
        let limits = ResolvedLimits::process_default();
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::with_limits(s, &limits)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    if !super::supports_compiled_measurement_sampling(circuit) {
        if let Ok(deferred) = defer_noisy_circuit(circuit, noise) {
            return run_deferred_shots_noisy(&deferred, num_shots, seed);
        }
        let limits = ResolvedLimits::process_default();
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::with_limits(s, &limits)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    if use_frame_sampler(circuit) {
        run_shots_noisy_frame(circuit, noise, num_shots, seed)
    } else {
        run_shots_noisy_compiled(circuit, noise, num_shots, seed)
    }
}

/// Sample a circuit [`defer_noisy_circuit`] rewrote: its terminal form through
/// [`run_validated_shots_noisy`] when no guarded body carries an event, else through
/// [`run_shots_noisy_guarded`].
pub(crate) fn run_deferred_shots_noisy(
    deferred: &DeferredCircuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    if deferred.guarded.is_empty() {
        return run_validated_shots_noisy(&deferred.circuit, &deferred.noise, num_shots, seed);
    }
    run_shots_noisy_guarded(deferred, num_shots, seed)
}

#[cfg(feature = "gpu")]
pub(crate) fn run_shots_noisy_with_gpu(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    context: std::sync::Arc<crate::gpu::GpuContext>,
) -> Result<ShotsResult> {
    noise.ensure_only_pauli_channels()?;
    if !circuit.is_clifford_only() {
        let limits = ResolvedLimits::process_default();
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::with_limits(s, &limits)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    if !super::supports_compiled_measurement_sampling(circuit) {
        if let Ok(deferred) = defer_noisy_circuit(circuit, noise) {
            if !deferred.guarded.is_empty() {
                return run_shots_noisy_guarded(&deferred, num_shots, seed);
            }
            return run_shots_noisy_with_gpu(
                &deferred.circuit,
                &deferred.noise,
                num_shots,
                seed,
                context,
            );
        }
        let limits = ResolvedLimits::process_default();
        return run_shots_noisy_brute_with(
            |s| Box::new(StabilizerBackend::with_limits(s, &limits)),
            circuit,
            noise,
            num_shots,
            seed,
        );
    }

    if use_frame_sampler(circuit) {
        return run_shots_noisy_frame(circuit, noise, num_shots, seed);
    }

    let sampler = compile_validated_noisy(circuit, noise, seed)?.with_gpu(context);
    finish_noisy_compiled_run(sampler, circuit, num_shots)
}

#[cfg(test)]
mod tests;
