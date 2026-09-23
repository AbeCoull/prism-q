//! Noise channels, noise models, and device calibration tables.

use num_complex::Complex64;
use prism_q::sim::calibration::presets;
use prism_q::{DeviceCalibration, NoiseChannel, NoiseEvent, NoiseModel};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use smallvec::SmallVec;

use crate::circuit::PyCircuit;
use crate::error::{PyPrismResult, invalid};
use crate::gate::extract_2x2;

/// A one- or two-qubit noise channel, built by the static methods.
#[pyclass(name = "NoiseChannel", module = "prism_q", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyNoiseChannel(pub NoiseChannel);

#[pymethods]
impl PyNoiseChannel {
    /// Independent Pauli X/Y/Z error with per-branch probabilities.
    #[staticmethod]
    fn pauli(px: f64, py: f64, pz: f64) -> Self {
        Self(NoiseChannel::Pauli { px, py, pz })
    }

    /// Symmetric depolarizing channel.
    #[staticmethod]
    fn depolarizing(p: f64) -> Self {
        Self(NoiseChannel::Depolarizing { p })
    }

    /// Amplitude damping (T1 relaxation).
    #[staticmethod]
    fn amplitude_damping(gamma: f64) -> Self {
        Self(NoiseChannel::AmplitudeDamping { gamma })
    }

    /// Pure dephasing.
    #[staticmethod]
    fn phase_damping(gamma: f64) -> Self {
        Self(NoiseChannel::PhaseDamping { gamma })
    }

    /// Combined T1 + T2 relaxation over a gate of duration `gate_time`.
    ///
    /// `excited_population` is the steady state the qubit relaxes toward: 0 for
    /// a cold qubit settling in the ground state, 0.5 for a maximally mixed
    /// steady state.
    #[staticmethod]
    #[pyo3(signature = (t1, t2, gate_time, excited_population = 0.0))]
    fn thermal_relaxation(t1: f64, t2: f64, gate_time: f64, excited_population: f64) -> Self {
        Self(NoiseChannel::ThermalRelaxation {
            t1,
            t2,
            gate_time,
            excited_population,
        })
    }

    /// Symmetric two-qubit depolarizing channel.
    #[staticmethod]
    fn two_qubit_depolarizing(p: f64) -> Self {
        Self(NoiseChannel::TwoQubitDepolarizing { p })
    }

    /// General single-qubit channel from a list of 2x2 complex Kraus operators.
    #[staticmethod]
    fn custom(kraus: Vec<Bound<'_, PyAny>>) -> PyPrismResult<Self> {
        let mats: Vec<[[Complex64; 2]; 2]> = kraus
            .iter()
            .map(extract_2x2)
            .collect::<PyPrismResult<_>>()?;
        if mats.is_empty() {
            return Err(invalid(
                "custom channel requires at least one Kraus operator",
            ));
        }
        Ok(Self(NoiseChannel::Custom { kraus: mats }))
    }

    fn __repr__(&self) -> String {
        format!("NoiseChannel({:?})", self.0)
    }
}

/// A noise model: per-instruction channels plus optional readout error.
#[pyclass(name = "NoiseModel", module = "prism_q")]
pub struct PyNoiseModel {
    pub inner: NoiseModel,
}

#[pymethods]
impl PyNoiseModel {
    /// Uniform single-qubit depolarizing noise after every gate.
    #[staticmethod]
    fn uniform_depolarizing(circuit: &PyCircuit, p: f64) -> Self {
        Self {
            inner: NoiseModel::uniform_depolarizing(circuit.inner(), p),
        }
    }

    /// Amplitude damping after every gate.
    #[staticmethod]
    fn with_amplitude_damping(circuit: &PyCircuit, gamma: f64) -> Self {
        Self {
            inner: NoiseModel::with_amplitude_damping(circuit.inner(), gamma),
        }
    }

    /// An empty model sized to `circuit`; populate with `add_event`.
    #[staticmethod]
    fn empty(circuit: &PyCircuit) -> Self {
        let c = circuit.inner();
        Self {
            inner: NoiseModel {
                after_gate: vec![Vec::new(); c.instructions.len()],
                readout: vec![None; c.num_classical_bits],
            },
        }
    }

    /// Attach a channel that fires after the instruction at `instruction_index`.
    fn add_event(
        &mut self,
        instruction_index: usize,
        channel: &PyNoiseChannel,
        qubits: Vec<usize>,
    ) -> PyPrismResult<()> {
        let len = self.inner.after_gate.len();
        if instruction_index >= len {
            return Err(invalid(format!(
                "instruction_index {instruction_index} out of range (model has {len} instructions)"
            )));
        }
        let qubits: SmallVec<[usize; 2]> = qubits.into_iter().collect();
        self.inner.after_gate[instruction_index].push(NoiseEvent {
            channel: channel.0.clone(),
            qubits,
        });
        Ok(())
    }

    /// Set the same readout error on every classical bit.
    fn with_readout_error(&mut self, p01: f64, p10: f64) {
        self.inner.with_readout_error(p01, p10);
    }

    /// Validate channel probabilities and Kraus operators.
    fn validate(&self) -> PyPrismResult<()> {
        self.inner.validate()?;
        Ok(())
    }

    /// Whether the model holds only single-qubit Pauli channels and nothing
    /// else that can flip a bit. A live `two_qubit_depolarizing` or readout
    /// entry answers False and still runs on the stabilizer samplers.
    fn is_pauli_only(&self) -> bool {
        self.inner.is_pauli_only()
    }
}

/// A device calibration table: per-qubit coherence times and readout rates,
/// per-family gate durations and error rates. Lowered onto a circuit by
/// `to_noise_model`.
#[pyclass(name = "DeviceCalibration", module = "prism_q", frozen)]
pub struct PyDeviceCalibration(DeviceCalibration);

#[pymethods]
impl PyDeviceCalibration {
    /// Parse the line-oriented text form: `qubit <n> t1= t2= p01= p10=`,
    /// `gate1q time= error=`, `gate2q time= error=`, `gate2q <a> <b> time= error=`.
    #[staticmethod]
    fn parse(text: &str) -> PyPrismResult<Self> {
        Ok(Self(DeviceCalibration::parse(text)?))
    }

    /// Illustrative transmon magnitudes, the same on every qubit.
    #[staticmethod]
    fn superconducting_transmon(num_qubits: usize) -> Self {
        Self(presets::superconducting_transmon(num_qubits))
    }

    /// Illustrative trapped-ion magnitudes, the same on every qubit.
    #[staticmethod]
    fn trapped_ion(num_qubits: usize) -> Self {
        Self(presets::trapped_ion(num_qubits))
    }

    /// Illustrative neutral-atom magnitudes, the same on every qubit.
    #[staticmethod]
    fn neutral_atom(num_qubits: usize) -> Self {
        Self(presets::neutral_atom(num_qubits))
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }

    /// Thermal relaxation and depolarizing after every gate, readout error on
    /// every measured bit.
    fn to_noise_model(&self, circuit: &PyCircuit) -> PyPrismResult<PyNoiseModel> {
        Ok(PyNoiseModel {
            inner: self.0.to_noise_model(circuit.inner())?,
        })
    }

    fn __repr__(&self) -> String {
        format!("DeviceCalibration(num_qubits={})", self.0.num_qubits())
    }
}

impl PyNoiseModel {
    /// Deep-copy the inner model, so a simulation can own it with the GIL released.
    pub fn clone_model(&self) -> NoiseModel {
        NoiseModel {
            after_gate: self.inner.after_gate.clone(),
            readout: self.inner.readout.clone(),
        }
    }
}
