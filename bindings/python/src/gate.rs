//! `Gate` wrapper exposing only the safe, user-constructible gate set.
//!
//! The fusion-internal variants (`Fused`, `BatchPhase`, `MultiFused`, ...) are
//! never exposed: they carry internal qubit indices and constructing them by
//! hand corrupts simulation state.

use num_complex::Complex64;
use prism_q::Gate;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::error::{PyPrismResult, invalid};

/// A quantum gate. Construct via the named static methods (`Gate.h()`,
/// `Gate.rx(theta)`, `Gate.cu(matrix)`, ...).
#[pyclass(name = "Gate", module = "prism_q", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyGate(pub Gate);

/// Extract a 2x2 complex matrix from a nested Python sequence or NumPy array.
pub fn extract_2x2(obj: &Bound<'_, PyAny>) -> PyPrismResult<[[Complex64; 2]; 2]> {
    let rows: Vec<Vec<Complex64>> = obj
        .extract()
        .map_err(|_| invalid("expected a 2x2 complex matrix (nested sequence or ndarray)"))?;
    if rows.len() != 2 || rows.iter().any(|r| r.len() != 2) {
        return Err(invalid(format!(
            "matrix must be 2x2, got {} rows",
            rows.len()
        )));
    }
    Ok([[rows[0][0], rows[0][1]], [rows[1][0], rows[1][1]]])
}

#[pymethods]
impl PyGate {
    #[staticmethod]
    fn id() -> Self {
        Self(Gate::Id)
    }
    #[staticmethod]
    fn x() -> Self {
        Self(Gate::X)
    }
    #[staticmethod]
    fn y() -> Self {
        Self(Gate::Y)
    }
    #[staticmethod]
    fn z() -> Self {
        Self(Gate::Z)
    }
    #[staticmethod]
    fn h() -> Self {
        Self(Gate::H)
    }
    #[staticmethod]
    fn s() -> Self {
        Self(Gate::S)
    }
    #[staticmethod]
    fn sdg() -> Self {
        Self(Gate::Sdg)
    }
    #[staticmethod]
    fn t() -> Self {
        Self(Gate::T)
    }
    #[staticmethod]
    fn tdg() -> Self {
        Self(Gate::Tdg)
    }
    #[staticmethod]
    fn sx() -> Self {
        Self(Gate::SX)
    }
    #[staticmethod]
    fn sxdg() -> Self {
        Self(Gate::SXdg)
    }
    #[staticmethod]
    fn cx() -> Self {
        Self(Gate::Cx)
    }
    #[staticmethod]
    fn cz() -> Self {
        Self(Gate::Cz)
    }
    #[staticmethod]
    fn swap() -> Self {
        Self(Gate::Swap)
    }

    #[staticmethod]
    fn rx(theta: f64) -> Self {
        Self(Gate::Rx(theta))
    }
    #[staticmethod]
    fn ry(theta: f64) -> Self {
        Self(Gate::Ry(theta))
    }
    #[staticmethod]
    fn rz(theta: f64) -> Self {
        Self(Gate::Rz(theta))
    }
    #[staticmethod]
    fn p(theta: f64) -> Self {
        Self(Gate::P(theta))
    }
    #[staticmethod]
    fn rzz(theta: f64) -> Self {
        Self(Gate::Rzz(theta))
    }

    /// Controlled phase `diag(1, e^{i*theta})` on [control, target].
    #[staticmethod]
    fn cphase(theta: f64) -> Self {
        Self(Gate::cphase(theta))
    }

    /// Controlled-unitary with a 2x2 target matrix on [control, target].
    #[staticmethod]
    fn cu(matrix: &Bound<'_, PyAny>) -> PyPrismResult<Self> {
        Ok(Self(Gate::cu(extract_2x2(matrix)?)))
    }

    /// Multi-controlled unitary: 2x2 target matrix gated on `num_controls` controls.
    #[staticmethod]
    fn mcu(matrix: &Bound<'_, PyAny>, num_controls: u8) -> PyPrismResult<Self> {
        if num_controls < 1 {
            return Err(invalid("mcu requires at least one control qubit"));
        }
        Ok(Self(Gate::mcu(extract_2x2(matrix)?, num_controls)))
    }

    /// Number of qubits the gate acts on.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }

    /// Canonical gate name.
    #[getter]
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn __repr__(&self) -> String {
        format!("Gate.{}", self.0.name())
    }
}

impl PyGate {
    pub fn inner(&self) -> &Gate {
        &self.0
    }
}
