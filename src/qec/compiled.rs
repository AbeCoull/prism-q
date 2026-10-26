//! Compile-once sampling of a native QEC program: [`QecCompiledSampler`].

use super::leakage::QecLeakageSampler;
use super::noise::{
    QecCompiledNoiseSampler, QecFeedforwardFlips, QecParityNoise, compile_qec_noisy_sampler,
    lower_qec_program_to_deferred_circuit,
};
use super::parity_walk::compile_parity_noise;
use super::reference::{
    QecReference, forced_tableau_records, forced_zero_records, has_entangled_reset,
};
use super::runner::{
    QecParityProjection, QecResultShape, lower_qec_program_to_clifford_circuit, qec_join,
    qec_records_unobserved, qec_result_from_measurement_chunks, qec_result_from_measurement_rows,
    qec_result_from_parity_projection, qec_runner_chunk_size, validate_qec_compiled_program,
};
use super::{QecDetectorReference, QecProgram, QecSampleResult};
use crate::error::{PrismError, Result};
use crate::sim::compiled::{
    CompiledSampler, PackedShots, compile_measurements, defer_measure_reset_circuit,
};
use crate::sim::splitmix64;

/// A native QEC program compiled once for the packed sampler, sampled any number of
/// times by [`Self::sample`].
///
/// Compiling does the work [`run_qec_program`](super::run_qec_program) repeats on every
/// call: lowering, the parity walk, the sampler tableau, and the noiseless reference of
/// [`QecOptions::detector_reference`](super::QecOptions::detector_reference). Shot
/// count and seed are per call; `chunk_size`, `keep_measurements`, and the reference
/// convention are fixed by the program's options at compile time. Sampling takes
/// `&mut self`, so share a sampler across threads behind a lock or compile one per
/// thread.
pub struct QecCompiledSampler {
    route: QecSamplerRoute,
    num_measurements: usize,
    detector_rows: Vec<Vec<usize>>,
    observable_rows: Vec<Vec<usize>>,
    postselection_rows: Vec<(Vec<usize>, bool)>,
    chunk_size: Option<usize>,
    keep_measurements: bool,
    reference: Option<QecReference>,
    next_seed: u64,
}

enum QecSamplerRoute {
    Empty,
    /// Detector and observable bits drawn without measurement records; the pattern is
    /// inverted into raw parity and skipped when results are referenced.
    Parity {
        projection: QecParityProjection,
        pattern: Option<Vec<u64>>,
        noise: Option<Box<QecParityNoise>>,
    },
    Records(Box<QecCompiledNoiseSampler>),
    Clean(Box<CompiledSampler>),
    Leaky(Box<QecLeakageSampler>),
}

impl QecProgram {
    /// Compile the program for repeated sampling through the packed Clifford path.
    ///
    /// Accepts what [`run_qec_program`](super::run_qec_program) runs on that path:
    /// Clifford gates, noise and leakage annotations, postselection, Pauli
    /// feed-forward, detectors, and observables. A program with `EXP_VAL` ops, a
    /// non-Clifford gate, or a feed-forward body with other gates is rejected; those run
    /// through `run_qec_program` or [`run_qec_program_reference`](super::run_qec_program_reference).
    pub fn compile_sampler(&self) -> Result<QecCompiledSampler> {
        super::validate_measured_qubit_reuse(self)?;
        if self.num_expectation_values() > 0 {
            return Err(PrismError::IncompatibleBackend {
                backend: "QEC compiled sampler".to_string(),
                reason: "the packed sampler has no estimator stage for `EXP_VAL`; \
                         `run_qec_program` routes such programs to the analytical or \
                         reference runners"
                    .to_string(),
            });
        }
        let (has_noise, has_feedforward) = validate_qec_compiled_program(self)?;
        let options = self.options();
        qec_runner_chunk_size(options)?;
        let referenced = options.detector_reference == QecDetectorReference::Noiseless;
        let detector_rows = self.detector_rows()?;
        let observable_rows = self.observable_rows()?;

        let mut sampler = QecCompiledSampler {
            route: QecSamplerRoute::Empty,
            num_measurements: self.num_measurements(),
            postselection_rows: self.postselection_rows()?,
            detector_rows,
            observable_rows,
            chunk_size: options.chunk_size,
            keep_measurements: options.keep_measurements,
            reference: None,
            next_seed: options.seed,
        };
        if self.has_leakage() {
            let mut leaky = QecLeakageSampler::compile(self)?;
            let (uncorrected, reference) =
                sampler.forced_zero_reference(self, leaky.noiseless(), None, referenced)?;
            leaky.noiseless_mut().rebase_reference(uncorrected);
            sampler.reference = reference;
            sampler.route = QecSamplerRoute::Leaky(Box::new(leaky));
            return Ok(sampler);
        }
        if self.num_measurements() == 0 {
            return Ok(sampler);
        }
        if !has_feedforward && qec_records_unobserved(self) {
            let (projection, deferred) = qec_join(
                || {
                    QecParityProjection::new(
                        self.num_measurements(),
                        &sampler.detector_rows,
                        &sampler.observable_rows,
                    )
                },
                || lower_qec_program_to_deferred_circuit(self),
            );
            if let Some((pattern, noise)) =
                compile_parity_noise(&deferred?, &projection, options.seed, None)?
            {
                sampler.route = QecSamplerRoute::Parity {
                    projection,
                    pattern: (!referenced).then_some(pattern),
                    noise: has_noise.then(|| Box::new(noise)),
                };
                return Ok(sampler);
            }
        }
        if has_noise || has_feedforward {
            let mut records = compile_qec_noisy_sampler(self)?;
            let (uncorrected, reference) = sampler.forced_zero_reference(
                self,
                records.noiseless(),
                records.feedforward(),
                referenced,
            )?;
            records.noiseless_mut().rebase_reference(uncorrected);
            sampler.reference = reference;
            sampler.route = QecSamplerRoute::Records(Box::new(records));
            return Ok(sampler);
        }

        let circuit = lower_qec_program_to_clifford_circuit(self)?;
        let circuit = if circuit.has_resets() || !circuit.has_terminal_measurements_only() {
            defer_measure_reset_circuit(&circuit)?
        } else {
            circuit
        };
        let mut clean = compile_measurements(&circuit, options.seed)?;
        let (uncorrected, reference) =
            sampler.forced_zero_reference(self, &clean, None, referenced)?;
        clean.rebase_reference(uncorrected);
        sampler.reference = reference;
        sampler.route = QecSamplerRoute::Clean(Box::new(clean));
        Ok(sampler)
    }
}

impl QecCompiledSampler {
    /// The forced-zero point of `noiseless` before `corrections`, which the sampler
    /// rebases onto so its draws depend on the seed alone, and the reference results are
    /// measured against, `None` for raw parity. The reference run's records are the
    /// corrected forced-zero point unless a reset collapses a qubit that later records
    /// depend on, in which case the tableau run decides them.
    fn forced_zero_reference(
        &self,
        program: &QecProgram,
        noiseless: &CompiledSampler,
        corrections: Option<&QecFeedforwardFlips>,
        referenced: bool,
    ) -> Result<(Vec<u64>, Option<QecReference>)> {
        let (uncorrected, corrected) = forced_zero_records(noiseless, corrections);
        if !referenced || self.detector_rows.len() + self.observable_rows.len() == 0 {
            return Ok((uncorrected, None));
        }
        let records = if has_entangled_reset(program) {
            forced_tableau_records(program)?
                .expect("the packed path admits Clifford gates only, so the tableau run exists")
        } else {
            corrected
        };
        let reference =
            QecReference::from_records(&records, &self.detector_rows, &self.observable_rows);
        Ok((uncorrected, Some(reference)))
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn num_detectors(&self) -> usize {
        self.detector_rows.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observable_rows.len()
    }

    /// Sample `shots` shots.
    ///
    /// `Some(seed)` draws exactly what [`run_qec_program`](super::run_qec_program) draws
    /// for a program with that seed and `shots`. `None` continues the sampler's stream:
    /// the first call draws with the program's seed, and each call after one drawn with
    /// seed `s` draws with a seed derived from `s`, so calls are independent and the
    /// sequence repeats for a fixed program seed.
    pub fn sample(&mut self, shots: usize, seed: Option<u64>) -> Result<QecSampleResult> {
        let seed = seed.unwrap_or(self.next_seed);
        self.next_seed = splitmix64(seed);
        let chunk_size = self.chunk_size.unwrap_or(shots.max(1));
        let shape = QecResultShape {
            num_measurements: self.num_measurements,
            keep_measurements: self.keep_measurements,
            detector_rows: &self.detector_rows,
            observable_rows: &self.observable_rows,
            postselection_rows: &self.postselection_rows,
            reference: self.reference.as_ref(),
        };
        match &mut self.route {
            QecSamplerRoute::Empty => qec_result_from_measurement_rows(
                shape,
                shots,
                PackedShots::from_shot_major(Vec::new(), shots, self.num_measurements),
            ),
            QecSamplerRoute::Parity {
                projection,
                pattern,
                noise,
            } => {
                if let Some(noise) = noise.as_mut() {
                    noise.reseed(seed);
                }
                qec_result_from_parity_projection(
                    shots,
                    self.num_measurements,
                    projection,
                    pattern.as_deref(),
                    noise.as_deref(),
                )
            }
            QecSamplerRoute::Records(sampler) => {
                sampler.reseed(seed);
                if chunk_size >= shots {
                    let measurements = sampler.sample_measurements_packed(0, shots, shots)?;
                    return qec_result_from_measurement_rows(shape, shots, measurements);
                }
                qec_result_from_measurement_chunks(shape, shots, chunk_size, |first, chunk| {
                    sampler.sample_measurements_packed(first, chunk, shots)
                })
            }
            QecSamplerRoute::Clean(sampler) => {
                sampler.reseed(seed);
                if chunk_size >= shots {
                    let measurements = sampler.try_sample_bulk_packed(shots)?;
                    return qec_result_from_measurement_rows(shape, shots, measurements);
                }
                qec_result_from_measurement_chunks(shape, shots, chunk_size, |_, chunk| {
                    sampler.try_sample_bulk_packed(chunk)
                })
            }
            QecSamplerRoute::Leaky(sampler) => {
                sampler.reseed(seed);
                sample_leaky(sampler, shape, shots, chunk_size)
            }
        }
    }
}

/// The packed path for a program carrying leakage annotations, which a linear map of
/// record flips cannot express: [`QecLeakageSampler`] carries the flags, and its herald
/// columns land in [`QecSampleResult::heralds`].
fn sample_leaky(
    sampler: &mut QecLeakageSampler,
    shape: QecResultShape<'_>,
    shots: usize,
    chunk_size: usize,
) -> Result<QecSampleResult> {
    if chunk_size >= shots {
        let (measurements, heralds) = sampler.sample(0, shots, shots)?;
        return Ok(
            qec_result_from_measurement_rows(shape, shots, measurements)?.with_heralds(heralds),
        );
    }
    let num_heralds = sampler.num_heralds();
    let mut herald_chunks: Vec<(usize, PackedShots)> = Vec::new();
    let result = qec_result_from_measurement_chunks(shape, shots, chunk_size, |first, chunk| {
        let (measurements, heralds) = sampler.sample(first, chunk, shots)?;
        herald_chunks.push((first, heralds));
        Ok(measurements)
    })?;
    let s_words = shots.div_ceil(64);
    let mut heralds = vec![0u64; num_heralds * s_words];
    for (first_shot, chunk) in herald_chunks {
        let chunk_words = chunk.num_shots().div_ceil(64);
        for herald in 0..num_heralds {
            super::leakage::land_bits(
                &mut heralds[herald * s_words..(herald + 1) * s_words],
                first_shot,
                &chunk.raw_data()[herald * chunk_words..(herald + 1) * chunk_words],
                0,
                chunk.num_shots(),
                |a, b| a | b,
            );
        }
    }
    Ok(result.with_heralds(PackedShots::from_meas_major(heralds, shots, num_heralds)))
}

#[cfg(test)]
#[path = "compiled_tests.rs"]
mod tests;
