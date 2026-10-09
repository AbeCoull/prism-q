//! PyO3 bindings over the public `prism-q` API. The compiled module is
//! `prism_q._prism_q`, re-exported by the pure-Python `prism_q` package.

use pyo3::prelude::*;

mod backend;
mod braket;
mod circuit;
mod codec;
mod distributed;
mod error;
mod gate;
mod gpu;
mod noise;
mod numpy_util;
mod parameter;
mod pickle;
mod program;
mod qec;
mod sampler;
mod sim;

use backend::{PyBackendKind, PyStabilizerBackend};
use braket::PyBraketProgram;
use circuit::{PyCircuit, PyCircuitBuilder, PyClassicalCondition, PySaveSpec};
use error::PrismError;
use gate::PyGate;
use gpu::{PyGpuContext, PyGpuInfo};
use noise::{
    PyDeviceCalibration, PyErrorChainComplex, PyGateFilter, PyNoiseBuilder, PyNoiseChannel,
    PyNoiseModel,
};
use parameter::{PyParameters, PyPreparedCircuit};
use program::{PyDynamicProgram, PyDynamicProgramBuilder, PyProgramSimulation};
use qec::{
    PyDecoder, PyDetectorErrorModel, PyQecBasis, PyQecNoise, PyQecProgram, PyQecResult, PyRecordRef,
};
use sampler::PyCompiledSampler;
use sim::{
    PyBondReport, PyCountsResult, PyEntropyResult, PyExpectationResult, PyObservableExpectation,
    PyObservableVariance, PyOverlapResult, PyPauliObservable, PyReducedDensityMatrix,
    PyRunMetadata, PyRunOutcome, PyShotsResult, PySimulation,
};

#[pymodule]
fn _prism_q(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("PrismError", m.py().get_type::<PrismError>())?;

    m.add_class::<PyGate>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyCircuitBuilder>()?;
    m.add_class::<PySaveSpec>()?;
    m.add_class::<PyClassicalCondition>()?;
    m.add_class::<PyBackendKind>()?;
    m.add_class::<PyStabilizerBackend>()?;
    m.add_class::<PyParameters>()?;
    m.add_class::<PyPreparedCircuit>()?;
    m.add_class::<PyGpuContext>()?;
    m.add_class::<PyGpuInfo>()?;
    m.add_class::<crate::distributed::PyDistributedContext>()?;
    m.add_class::<PyNoiseChannel>()?;
    m.add_class::<PyGateFilter>()?;
    m.add_class::<PyNoiseBuilder>()?;
    m.add_class::<PyNoiseModel>()?;
    m.add_class::<PyDeviceCalibration>()?;
    m.add_class::<PyErrorChainComplex>()?;
    m.add_class::<PySimulation>()?;
    m.add_class::<PyRunOutcome>()?;
    m.add_class::<PyShotsResult>()?;
    m.add_class::<PyCountsResult>()?;
    m.add_class::<PyCompiledSampler>()?;
    m.add_class::<PyRunMetadata>()?;
    m.add_class::<PyBondReport>()?;
    m.add_class::<PyPauliObservable>()?;
    m.add_class::<PyObservableExpectation>()?;
    m.add_class::<PyObservableVariance>()?;
    m.add_class::<PyEntropyResult>()?;
    m.add_class::<PyExpectationResult>()?;
    m.add_class::<PyOverlapResult>()?;
    m.add_class::<PyReducedDensityMatrix>()?;
    m.add_class::<PyQecBasis>()?;
    m.add_class::<PyRecordRef>()?;
    m.add_class::<PyQecNoise>()?;
    m.add_class::<PyQecProgram>()?;
    m.add_class::<PyQecResult>()?;
    m.add_class::<PyDetectorErrorModel>()?;
    m.add_class::<PyDecoder>()?;
    m.add_class::<PyBraketProgram>()?;
    m.add_class::<PyDynamicProgram>()?;
    m.add_class::<PyDynamicProgramBuilder>()?;
    m.add_class::<PyProgramSimulation>()?;

    m.add_function(wrap_pyfunction!(circuit::parse_qasm, m)?)?;
    m.add_function(wrap_pyfunction!(circuit::parse_qasm_parametric, m)?)?;
    m.add_function(wrap_pyfunction!(braket::parse_braket, m)?)?;
    m.add_function(wrap_pyfunction!(program::parse_qasm_dynamic, m)?)?;
    m.add_function(wrap_pyfunction!(program::simulate_program, m)?)?;
    m.add_function(wrap_pyfunction!(sim::simulate, m)?)?;
    m.add_function(wrap_pyfunction!(sim::run_qasm, m)?)?;
    m.add_function(wrap_pyfunction!(sim::run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(gpu::gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(noise::noisy_marginals_analytical, m)?)?;

    circuit::register_circuits(m)?;

    Ok(())
}
