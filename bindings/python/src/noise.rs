//! Noise channels, model construction rules, and device calibration tables.

use num_complex::Complex64;
use numpy::PyArray1;
use prism_q::sim::calibration::presets;
use prism_q::{
    DeviceCalibration, DriftDistribution, ErrorChainComplex, GateFilter, GateTimes, NoiseBuilder,
    NoiseChannel, NoiseEvent, NoiseModel, PauliAxis,
};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use smallvec::SmallVec;

use crate::circuit::PyCircuit;
use crate::codec::{self, Kind, Reader, Writer};
use crate::error::{PyPrismResult, invalid};
use crate::gate::{extract_2x2, extract_4x4};
use crate::numpy_util::f64_array;
use crate::pickle::{Reduced, reduce};

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

    /// General two-qubit channel from a list of 4x4 complex Kraus operators,
    /// indexed with the first target qubit as the high bit of both indices.
    #[staticmethod]
    fn custom_2q(kraus: Vec<Bound<'_, PyAny>>) -> PyPrismResult<Self> {
        let mats: Vec<[[Complex64; 4]; 4]> = kraus
            .iter()
            .map(extract_4x4)
            .collect::<PyPrismResult<_>>()?;
        if mats.is_empty() {
            return Err(invalid(
                "custom_2q channel requires at least one Kraus operator",
            ));
        }
        Ok(Self(NoiseChannel::Kraus2q { kraus: mats }))
    }

    /// Number of target qubits an event carrying this channel names.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }

    #[staticmethod]
    fn _from_pickle(data: &[u8]) -> PyPrismResult<Self> {
        let mut r = Reader::new(data, Kind::NoiseChannel)?;
        let channel = codec::read_channel(&mut r)?;
        r.finish()?;
        Ok(Self(channel))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduced<'py>> {
        let mut w = Writer::new(Kind::NoiseChannel);
        codec::write_channel(&mut w, &slf.get().0)?;
        reduce(slf.as_any(), w.finish())
    }

    /// An unleaked qubit leaves the computational subspace with probability `p`.
    #[staticmethod]
    fn leakage(p: f64) -> Self {
        Self(NoiseChannel::Leakage { p })
    }

    /// A leaked qubit returns in a uniformly random basis state with probability `p`.
    #[staticmethod]
    fn seepage(p: f64) -> Self {
        Self(NoiseChannel::Seepage { p })
    }

    /// Two-qubit: when exactly one qubit is leaked, the other leaks with probability `p`.
    #[staticmethod]
    fn leakage_transport(p: f64) -> Self {
        Self(NoiseChannel::LeakageTransport { p })
    }

    /// Rotation about `axis` (`"x"`, `"y"` or `"z"`) by `sum(w * z[source])` over
    /// `(source, w)` pairs, `z` standard normal draws made once per shot.
    #[staticmethod]
    fn quasi_static(axis: &str, weights: Vec<(usize, f64)>) -> PyPrismResult<Self> {
        let axis = match axis {
            "x" | "X" => PauliAxis::X,
            "y" | "Y" => PauliAxis::Y,
            "z" | "Z" => PauliAxis::Z,
            other => return Err(invalid(format!("axis must be x, y or z, got {other:?}"))),
        };
        Ok(Self(NoiseChannel::QuasiStatic { axis, weights }))
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            NoiseChannel::Custom { kraus } => {
                format!("NoiseChannel.custom({} operators)", kraus.len())
            }
            NoiseChannel::Kraus2q { kraus } => {
                format!("NoiseChannel.custom_2q({} operators)", kraus.len())
            }
            other => format!("NoiseChannel({other:?})"),
        }
    }
}

/// Instruction durations in seconds for timing an as-soon-as-possible schedule.
#[pyclass(name = "GateTimes", module = "prism_q", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyGateTimes(GateTimes);

#[pymethods]
impl PyGateTimes {
    /// Durations of one-qubit gates and of gates on two or more qubits.
    #[new]
    fn new(one_qubit: f64, two_qubit: f64) -> Self {
        Self(GateTimes::new(one_qubit, two_qubit))
    }

    /// Give gates named `name` their own duration; `"id"` makes identities delays.
    fn with_gate(mut slf: PyRefMut<'_, Self>, name: String, time: f64) -> PyRefMut<'_, Self> {
        slf.0 = slf.0.clone().with_gate(name, time);
        slf
    }

    fn with_pair(mut slf: PyRefMut<'_, Self>, a: usize, b: usize, time: f64) -> PyRefMut<'_, Self> {
        slf.0 = slf.0.clone().with_pair(a, b, time);
        slf
    }

    fn with_measure(mut slf: PyRefMut<'_, Self>, time: f64) -> PyRefMut<'_, Self> {
        slf.0 = slf.0.clone().with_measure(time);
        slf
    }

    fn with_reset(mut slf: PyRefMut<'_, Self>, time: f64) -> PyRefMut<'_, Self> {
        slf.0 = slf.0.clone().with_reset(time);
        slf
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Zero-mean Gaussian over one quasi-static offset per qubit, drawn once per shot.
#[pyclass(name = "DriftDistribution", module = "prism_q", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDriftDistribution(DriftDistribution);

#[pymethods]
impl PyDriftDistribution {
    #[staticmethod]
    fn independent(sigmas: Vec<f64>) -> Self {
        Self(DriftDistribution::independent(sigmas))
    }

    /// Detuning widths `sqrt(2) / t2_star` in radians per second.
    #[staticmethod]
    fn from_t2_star(t2_star: Vec<f64>) -> Self {
        Self(DriftDistribution::from_t2_star(t2_star))
    }

    #[staticmethod]
    fn from_covariance(matrix: Vec<Vec<f64>>) -> Self {
        Self(DriftDistribution::from_covariance(matrix))
    }

    /// Correlate each undirected coupling edge with coefficient `rho`.
    fn with_neighbour_correlation(
        mut slf: PyRefMut<'_, Self>,
        coupling: Vec<(usize, usize)>,
        rho: f64,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let n = slf.0.num_qubits();
        if let Some(&(a, b)) = coupling.iter().find(|&&(a, b)| a >= n || b >= n) {
            return Err(invalid(format!(
                "coupling edge ({a}, {b}) is outside the {n}-qubit drift distribution"
            )));
        }
        slf.0 = slf.0.clone().with_neighbour_correlation(coupling, rho);
        Ok(slf)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }

    /// Covariance as a list of rows.
    fn covariance(&self) -> Vec<Vec<f64>> {
        let n = self.0.num_qubits();
        self.0
            .covariance()
            .chunks(n.max(1))
            .map(<[f64]>::to_vec)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("DriftDistribution(num_qubits={})", self.0.num_qubits())
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

    /// Time the circuit for `scheduled_idle` and `quasi_static_detuning`.
    fn schedule<'py>(mut slf: PyRefMut<'py, Self>, times: &PyGateTimes) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).schedule(times.0.clone());
        slf
    }

    /// Relax each qubit over the idle part of every scheduled layer from its `(t1, t2)`.
    fn scheduled_idle(
        mut slf: PyRefMut<'_, Self>,
        coherence: Vec<(f64, f64)>,
    ) -> PyRefMut<'_, Self> {
        slf.0 = std::mem::take(&mut slf.0).scheduled_idle(coherence);
        slf
    }

    /// Per-shot detuning in rad/s, applied as a Z phase over each scheduled layer.
    fn quasi_static_detuning<'py>(
        mut slf: PyRefMut<'py, Self>,
        drift: &PyDriftDistribution,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).quasi_static_detuning(drift.0.clone());
        slf
    }

    /// One per-shot fractional angle error with deviation `sigma` shared by matching rotations.
    fn over_rotation_drift<'py>(
        mut slf: PyRefMut<'py, Self>,
        filter: &PyGateFilter,
        sigma: f64,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0).over_rotation_drift(filter.0.clone(), sigma);
        slf
    }

    /// Per-shot fractional angle errors per target qubit, drawn jointly from `drift`.
    fn over_rotation_drift_per_qubit<'py>(
        mut slf: PyRefMut<'py, Self>,
        filter: &PyGateFilter,
        drift: &PyDriftDistribution,
    ) -> PyRefMut<'py, Self> {
        slf.0 = std::mem::take(&mut slf.0)
            .over_rotation_drift_per_qubit(filter.0.clone(), drift.0.clone());
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

    #[staticmethod]
    fn _from_pickle(data: &[u8]) -> PyPrismResult<Self> {
        Ok(Self {
            inner: codec::decode_noise_model(data)?,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduced<'py>> {
        let data = codec::encode_noise_model(&slf.borrow().inner)?;
        reduce(slf.as_any(), data)
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

    /// `to_noise_model` plus idle relaxation timed by the table's own gate durations.
    fn to_scheduled_noise_model(&self, circuit: &PyCircuit) -> PyPrismResult<PyNoiseModel> {
        Ok(PyNoiseModel {
            inner: self.0.to_scheduled_noise_model(circuit.inner())?,
        })
    }

    /// The table's gate durations, pair entries included.
    fn gate_times(&self) -> PyGateTimes {
        PyGateTimes(self.0.gate_times())
    }

    fn __repr__(&self) -> String {
        format!("DeviceCalibration(num_qubits={})", self.0.num_qubits())
    }
}

/// The GF(2) chain complex of a noisy Clifford circuit: which noise locations
/// flip which measurement records, and which error classes no measurement
/// detects.
#[pyclass(name = "ErrorChainComplex", module = "prism_q", frozen)]
pub struct PyErrorChainComplex {
    inner: ErrorChainComplex,
    num_measurements: usize,
}

#[pymethods]
impl PyErrorChainComplex {
    #[new]
    fn new(circuit: &PyCircuit, noise: &PyNoiseModel) -> PyPrismResult<Self> {
        let circuit = circuit.inner();
        noise.inner.validate_for(circuit)?;
        Ok(Self {
            inner: ErrorChainComplex::build(circuit, &noise.inner, 0)?,
            num_measurements: circuit.measurement_map().len(),
        })
    }

    /// Stabilizer generators no measurement detects.
    #[getter]
    fn boundary_dim(&self) -> usize {
        self.inner.boundary_dim()
    }

    /// Independent logical error classes.
    #[getter]
    fn homology_dim(&self) -> usize {
        self.inner.homology_dim()
    }

    /// Exact noisy probability that each measurement record reads 1, from the
    /// noiseless ones, with no sampling.
    fn noisy_marginals<'py>(
        &self,
        py: Python<'py>,
        noiseless: Vec<f64>,
    ) -> PyPrismResult<Bound<'py, PyArray1<f64>>> {
        if noiseless.len() != self.num_measurements {
            return Err(invalid(format!(
                "expected {} noiseless marginals, one per measurement record, got {}",
                self.num_measurements,
                noiseless.len()
            )));
        }
        Ok(f64_array(py, self.inner.noisy_marginals(&noiseless)))
    }

    fn __repr__(&self) -> String {
        format!(
            "ErrorChainComplex(boundary_dim={}, homology_dim={})",
            self.inner.boundary_dim(),
            self.inner.homology_dim()
        )
    }
}

/// Exact per-classical-bit probability of reading 1 under `noise`, readout
/// error included, computed from the chain complex with no sampling. The
/// circuit must be Clifford with terminal measurements and `noise` a Pauli
/// model; bits no measurement writes read 0.5.
#[pyfunction]
pub fn noisy_marginals_analytical<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    noise: &PyNoiseModel,
) -> PyPrismResult<Bound<'py, PyArray1<f64>>> {
    let circuit = circuit.inner();
    noise.inner.validate_for(circuit)?;
    let marginals = py.detach(|| prism_q::noisy_marginals_analytical(circuit, &noise.inner, 0))?;
    Ok(f64_array(py, marginals))
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
