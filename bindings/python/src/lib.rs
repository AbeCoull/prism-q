//! Python bindings for PRISM-Q.
//!
//! This crate is a thin PyO3 wrapper over the public `prism-q` API. The
//! compiled extension module is `prism_q._prism_q`; the pure-Python
//! `prism_q` package re-exports it.

use pyo3::prelude::*;

mod backend;
mod circuit;
mod error;
mod gate;
mod noise;
mod numpy_util;
mod qec;
mod sim;

use backend::PyBackendKind;
use circuit::{PyCircuit, PyCircuitBuilder};
use error::PrismError;
use gate::PyGate;
use noise::{PyNoiseChannel, PyNoiseModel};
use qec::{PyQecBasis, PyQecNoise, PyQecProgram, PyQecResult, PyRecordRef};
use sim::{PyCountsResult, PyRunOutcome, PyShotsResult, PySimulation};

#[pymodule]
fn _prism_q(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("PrismError", m.py().get_type::<PrismError>())?;

    m.add_class::<PyGate>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyCircuitBuilder>()?;
    m.add_class::<PyBackendKind>()?;
    m.add_class::<PyNoiseChannel>()?;
    m.add_class::<PyNoiseModel>()?;
    m.add_class::<PySimulation>()?;
    m.add_class::<PyRunOutcome>()?;
    m.add_class::<PyShotsResult>()?;
    m.add_class::<PyCountsResult>()?;
    m.add_class::<PyQecBasis>()?;
    m.add_class::<PyRecordRef>()?;
    m.add_class::<PyQecNoise>()?;
    m.add_class::<PyQecProgram>()?;
    m.add_class::<PyQecResult>()?;

    m.add_function(wrap_pyfunction!(circuit::parse_qasm, m)?)?;
    m.add_function(wrap_pyfunction!(sim::simulate, m)?)?;
    m.add_function(wrap_pyfunction!(sim::run_qasm, m)?)?;

    circuit::register_circuits(m)?;

    Ok(())
}
