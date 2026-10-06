//! Noise channels, model construction rules, and device calibration tables.

use num_complex::Complex64;
use prism_q::sim::calibration::presets;
use prism_q::{DeviceCalibration, GateFilter, NoiseBuilder, NoiseChannel, NoiseEvent, NoiseModel};
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

/// Unset criteria match every gate; fluent methods update the same filter.
#[pyclass(name = "GateFilter", module = "prism_q")]
pub(crate) struct PyGateFilter(GateFilter);

#[pymethods]
impl PyGateFilter {
    #[new]
    fn new() -> Self {
        Self(GateFilter::all())
    }

    #[staticmethod]
    fn all() -> Self {
        Self::new()
    }

    fn arity(mut slf: PyRefMut<'_, Self>, arity: usize) -> PyRefMut<'_, Self> {
        slf.0 = std::mem::take(&mut slf.0).arity(arity);
        slf
    }

    /// Match the unfused gate name, such as `"cx"`; unknown names match nothing.
    fn named(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.0 = std::mem::take(&mut slf.0).named(name);
        slf
    }

    /// Select gate targets eligible for a rule, as an unordered set.
    fn on_qubits(mut slf: PyRefMut<'_, Self>, qubits: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.0 = std::mem::take(&mut slf.0).on_qubits(qubits);
        slf
    }

    /// Match the complete target list in order, so `[0, 1]` excludes `cx(1, 0)`.
    fn on_targets(mut slf: PyRefMut<'_, Self>, targets: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.0 = std::mem::take(&mut slf.0).on_targets(targets);
        slf
    }
}

/// Rules are copied when added and emit events in registration order at `build`.
#[pyclass(name = "NoiseBuilder", module = "prism_q")]
pub(crate) struct PyNoiseBuilder(NoiseBuilder);

#[pymethods]
impl PyNoiseBuilder {
    #[new]
    fn new() -> Self {
        Self(NoiseBuilder::new())
    }

    /// Emit a single-qubit channel on each matching target of a matching gate.
    fn after_gates<'py>(
        mut slf: PyRefMut<'py, Self>,
        filter: &PyGateFilter,
        channel: &PyNoiseChannel,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).after_gates(filter.0.clone(), channel.0.clone());
        slf
    }

    /// Emit a channel on the whole target list when its arity matches the gate.
    fn after_gates_joint<'py>(
        mut slf: PyRefMut<'py, Self>,
        filter: &PyGateFilter,
        channel: &PyNoiseChannel,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).after_gates_joint(filter.0.clone(), channel.0.clone());
        slf
    }

    /// Treat coupling edges as undirected; two-qubit channels order target before spectator.
    fn crosstalk<'py>(
        mut slf: PyRefMut<'py, Self>,
        filter: &PyGateFilter,
        coupling: Vec<(usize, usize)>,
        channel: &PyNoiseChannel,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).crosstalk(filter.0.clone(), coupling, channel.0.clone());
        slf
    }

    /// Append a rotation of `relative * theta` after matching `rx`, `ry`, `rz`, or `p` gates.
    fn over_rotation<'py>(
        mut slf: PyRefMut<'py, Self>,
        filter: &PyGateFilter,
        relative: f64,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).over_rotation(filter.0.clone(), relative);
        slf
    }

    /// Charge every untouched qubit once per greedy circuit layer, at its last instruction.
    fn on_idle_qubits<'py>(
        mut slf: PyRefMut<'py, Self>,
        channel: &PyNoiseChannel,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).on_idle_qubits(channel.0.clone());
        slf
    }

    /// Emit a single-qubit channel after each reset.
    fn after_resets<'py>(
        mut slf: PyRefMut<'py, Self>,
        channel: &PyNoiseChannel,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).after_resets(channel.0.clone());
        slf
    }

    /// Damage the measured state; a measurement at instruction zero needs a preceding barrier.
    fn before_measurements<'py>(
        mut slf: PyRefMut<'py, Self>,
        channel: &PyNoiseChannel,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).before_measurements(channel.0.clone());
        slf
    }

    /// Override the uniform readout rates on one classical bit, regardless of rule order.
    fn readout_error(
        mut slf: PyRefMut<'_, Self>,
        bit: usize,
        p01: f64,
        p10: f64,
    ) -> PyRefMut<'_, Self> {
        slf.0 = std::mem::take(&mut slf.0).readout_error(bit, p01, p10);
        slf
    }

    /// Set readout rates on every classical bit, including bits not measured.
    fn uniform_readout_error(
        mut slf: PyRefMut<'_, Self>,
        p01: f64,
        p10: f64,
    ) -> PyRefMut<'_, Self> {
        slf.0 = std::mem::take(&mut slf.0).uniform_readout_error(p01, p10);
        slf
    }

    /// Validate and lower against the unfused circuit; the builder remains reusable.
    fn build(&self, circuit: &PyCircuit) -> PyPrismResult<PyNoiseModel> {
        Ok(PyNoiseModel {
            inner: self.0.build(circuit.inner())?,
        })
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
