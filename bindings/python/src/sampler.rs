//! Compiled Clifford samplers held across calls, with streaming reductions.

use std::sync::Mutex;

use numpy::{PyArray1, PyArray2};
use prism_q::{
    CompiledSampler, CorrelatorAccumulator, Instruction, MarginalsAccumulator,
    NoisyCompiledSampler, PackedShots, PauliExpectationAccumulator, ShotAccumulator,
    compile_measurements, compile_noisy,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::PyCircuit;
use crate::error::{PyPrismResult, invalid};
use crate::noise::PyNoiseModel;
use crate::numpy_util::f64_array;
use crate::qec::{packed_to_2d, packed_to_bytes};
use crate::sim::{DEFAULT_SEED, counts_to_dict};

enum Engine {
    Noiseless(Box<CompiledSampler>),
    Noisy(Box<NoisyCompiledSampler>),
}

impl Engine {
    fn packed(&mut self, shots: usize) -> PyPrismResult<PackedShots> {
        Ok(match self {
            Engine::Noiseless(sampler) => sampler.try_sample_bulk_packed(shots)?,
            Engine::Noisy(sampler) => sampler.try_sample_bulk_packed(shots)?,
        })
    }

    fn stream<A: ShotAccumulator>(&mut self, shots: usize, acc: &mut A) {
        match self {
            Engine::Noiseless(sampler) => sampler.sample_chunked(shots, acc),
            Engine::Noisy(sampler) => sampler.sample_chunked(shots, acc),
        }
    }
}

/// A Clifford circuit compiled once into a parity sampler, for repeated draws
/// without recompiling.
///
/// Columns and count keys are measurement records in circuit order: record `j`
/// is the `j`-th measurement, and `Circuit.measurement_map()[j]` names its qubit
/// and classical bit. Each call continues one seeded stream, so repeated calls
/// draw fresh shots and a sampler rebuilt with the same seed replays them.
#[pyclass(name = "CompiledSampler", module = "prism_q")]
pub struct PyCompiledSampler {
    // The samplers are `Send` but not `Sync`; the lock lets calls release the GIL.
    inner: Mutex<Engine>,
    num_measurements: usize,
    rank: Option<usize>,
}

impl PyCompiledSampler {
    /// Never held across a call into Python, so poisoning means a panic already
    /// unwound through a method and the sampler state is not worth recovering.
    fn locked(&self) -> std::sync::MutexGuard<'_, Engine> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn check_record(&self, record: usize) -> PyPrismResult<()> {
        if record >= self.num_measurements {
            return Err(invalid(format!(
                "measurement record {record} out of range (circuit has {} measurements)",
                self.num_measurements
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyCompiledSampler {
    /// Compile `circuit`, which must be Clifford with terminal measurements and
    /// no reset or classical condition. With `noise`, every channel must be a
    /// Pauli channel, and readout error is applied to the record.
    #[new]
    #[pyo3(signature = (circuit, seed = DEFAULT_SEED, noise = None))]
    fn new(
        py: Python<'_>,
        circuit: &PyCircuit,
        seed: u64,
        noise: Option<PyRef<'_, PyNoiseModel>>,
    ) -> PyPrismResult<Self> {
        let circuit = circuit.inner();
        let num_measurements = circuit
            .instructions
            .iter()
            .filter(|inst| matches!(inst, Instruction::Measure { .. }))
            .count();
        let noise = noise.map(|model| model.clone_model());
        let engine = py.detach(|| -> prism_q::Result<Engine> {
            Ok(match &noise {
                Some(model) => Engine::Noisy(Box::new(compile_noisy(circuit, model, seed)?)),
                None => Engine::Noiseless(Box::new(compile_measurements(circuit, seed)?)),
            })
        })?;
        let rank = match &engine {
            Engine::Noiseless(sampler) => Some(sampler.rank()),
            Engine::Noisy(_) => None,
        };
        Ok(Self {
            inner: Mutex::new(engine),
            num_measurements,
            rank,
        })
    }

    #[getter]
    fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    /// Independent random bits behind the noiseless record, so at most
    /// `2 ** rank` distinct outcomes. `None` for a noisy sampler.
    #[getter]
    fn rank(&self) -> Option<usize> {
        self.rank
    }

    /// Draw `shots` records as a `(shots, num_measurements)` bool array.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
    ) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        let packed = py.detach(|| self.locked().packed(shots))?;
        packed_to_2d(py, &packed)
    }

    /// Draw `shots` records eight to a byte, as `(shots, ceil(num_measurements /
    /// 8))` uint8 in little bit order, the layout `QecResult.packed_measurements`
    /// uses.
    fn sample_packed<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
    ) -> PyPrismResult<Bound<'py, PyArray2<u8>>> {
        let packed = py.detach(|| self.locked().packed(shots))?;
        packed_to_bytes(py, &packed)
    }

    /// Histogram of `shots` records keyed by bitstring, character `j` being
    /// record `j`. A noiseless sampler whose rank is small against the shot
    /// count draws the histogram in closed form, with no per-shot work.
    fn sample_counts<'py>(&self, py: Python<'py>, shots: usize) -> PyResult<Bound<'py, PyDict>> {
        let counts = py.detach(|| -> PyPrismResult<_> {
            Ok(match &mut *self.locked() {
                Engine::Noiseless(sampler) => sampler.try_sample_counts(shots)?,
                Engine::Noisy(sampler) => sampler.try_sample_counts(shots)?,
            })
        })?;
        counts_to_dict(py, &counts, self.num_measurements)
    }

    /// Fraction of `shots` in which each record reads 1, as a float64 array.
    /// Shots stream through in bounded chunks, so the count is not limited by
    /// memory.
    fn marginals<'py>(&self, py: Python<'py>, shots: usize) -> Bound<'py, PyArray1<f64>> {
        let mut acc = MarginalsAccumulator::new(self.num_measurements);
        py.detach(|| self.locked().stream(shots, &mut acc));
        f64_array(py, acc.marginals())
    }

    /// `<(-1)^(parity of the records in each row)>` over `shots`, one float64
    /// per row: the expectation of the Z-type Pauli product those records
    /// measure. Streams like `marginals`.
    fn parity_expectations<'py>(
        &self,
        py: Python<'py>,
        rows: Vec<Vec<usize>>,
        shots: usize,
    ) -> PyPrismResult<Bound<'py, PyArray1<f64>>> {
        for &record in rows.iter().flatten() {
            self.check_record(record)?;
        }
        let mut acc = PauliExpectationAccumulator::new(rows);
        py.detach(|| self.locked().stream(shots, &mut acc));
        Ok(f64_array(py, acc.expectations()))
    }

    /// `<Z_i Z_j>` over `shots` for each `(i, j)` record pair, `1 - 2 *
    /// P(bits differ)`. Streams like `marginals`.
    fn correlators<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(usize, usize)>,
        shots: usize,
    ) -> PyPrismResult<Bound<'py, PyArray1<f64>>> {
        for &(i, j) in &pairs {
            self.check_record(i)?;
            self.check_record(j)?;
        }
        let mut acc = CorrelatorAccumulator::new(pairs);
        py.detach(|| self.locked().stream(shots, &mut acc));
        Ok(f64_array(py, acc.correlators()))
    }

    /// Every outcome with its multiplicity over the `2 ** rank` equally likely
    /// assignments, keyed as `sample_counts` keys. `None` for a noisy sampler or
    /// past rank 25.
    fn exact_counts<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let counts = match &*self.locked() {
            Engine::Noiseless(sampler) => sampler.exact_counts(),
            Engine::Noisy(_) => None,
        };
        counts
            .map(|counts| counts_to_dict(py, &counts, self.num_measurements))
            .transpose()
    }

    fn __repr__(&self) -> String {
        let rank = self
            .rank
            .map_or_else(|| "None".to_string(), |rank| rank.to_string());
        format!(
            "CompiledSampler(num_measurements={}, rank={rank})",
            self.num_measurements
        )
    }
}
