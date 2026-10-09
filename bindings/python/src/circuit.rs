//! Circuit construction, OpenQASM parsing, and reusable circuit builders.

use prism_q::circuit::{openqasm, qasm_export};
use prism_q::{
    Circuit, CircuitBuilder, ClassicalCondition, Gate, Instruction, PauliTerm, SaveSpec,
    SvgOptions, TextOptions, circuits,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::codec::{self, Kind, Reader, Writer};
use crate::error::{PyPrismResult, invalid};
use crate::gate::PyGate;
use crate::parameter::PyParameters;
use crate::pickle::{Reduced, ReducedMember, reduce, reduce_member};
use crate::sim::{parse_axis, parse_pauli_string};

/// A quantum circuit. Construct via [`CircuitBuilder`], [`parse_qasm`], or one
/// of the reusable circuit generators under `prism_q.circuits`.
#[pyclass(name = "Circuit", module = "prism_q", from_py_object)]
#[derive(Clone)]
pub struct PyCircuit(pub Circuit);

fn check_qubit(
    num_qubits: usize,
    qubit: usize,
    label: impl std::fmt::Display,
) -> PyPrismResult<()> {
    if qubit >= num_qubits {
        return Err(invalid(format!(
            "{label} {qubit} out of range (circuit has {num_qubits} qubits)"
        )));
    }
    Ok(())
}

fn check_classical_bit(num_bits: usize, bit: usize) -> PyPrismResult<()> {
    if bit >= num_bits {
        return Err(invalid(format!(
            "classical bit {bit} out of range (circuit has {num_bits} classical bits)"
        )));
    }
    Ok(())
}

fn check_targets(num_qubits: usize, gate: &Gate, targets: &[usize]) -> PyPrismResult<()> {
    let expected = gate.num_qubits();
    if targets.len() != expected {
        return Err(invalid(format!(
            "gate `{}` expects {expected} qubits, got {}",
            gate.name(),
            targets.len()
        )));
    }
    for (idx, &target) in targets.iter().enumerate() {
        check_qubit(num_qubits, target, format!("target[{idx}]"))?;
    }
    Ok(())
}

/// Parse `(qubit, axis)` factors and reject what `Circuit::add_pauli_rotation`
/// panics on, so bad input reaches Python as a `PrismError`.
fn check_pauli_factors(
    num_qubits: usize,
    factors: Vec<(usize, String)>,
) -> PyPrismResult<Vec<PauliTerm>> {
    if factors.is_empty() {
        return Err(invalid("pauli rotation needs at least one factor"));
    }
    let terms = parse_pauli_string(&factors)?;
    for (idx, term) in terms.iter().enumerate() {
        check_qubit(num_qubits, term.qubit, format!("factor[{idx}] qubit"))?;
    }
    let mut seen: Vec<usize> = terms.iter().map(|t| t.qubit).collect();
    seen.sort_unstable();
    if let Some(pair) = seen.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(invalid(format!(
            "pauli rotation has duplicate factor on qubit {}",
            pair[0]
        )));
    }
    Ok(terms)
}

fn check_distinct(qubits: &[usize]) -> PyPrismResult<()> {
    for (idx, qubit) in qubits.iter().enumerate() {
        if qubits[..idx].contains(qubit) {
            return Err(invalid(format!("qubit {qubit} named twice")));
        }
    }
    Ok(())
}

fn check_mcu_targets(num_qubits: usize, controls: &[usize], target: usize) -> PyPrismResult<()> {
    if controls.is_empty() {
        return Err(invalid("mcu requires at least one control qubit"));
    }
    if controls.len() > u8::MAX as usize {
        return Err(invalid(format!(
            "mcu supports at most {} control qubits, got {}",
            u8::MAX,
            controls.len()
        )));
    }
    for (idx, &control) in controls.iter().enumerate() {
        check_qubit(num_qubits, control, format!("control[{idx}]"))?;
    }
    check_qubit(num_qubits, target, "target")?;
    Ok(())
}

const REPR_MAX_QUBITS: usize = 64;
const REPR_MAX_MOMENTS: usize = 200;

/// A runtime test on measured classical bits that guards a gate or region.
#[pyclass(
    name = "ClassicalCondition",
    module = "prism_q",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyClassicalCondition(pub ClassicalCondition);

/// Classical bits `condition` reads.
pub(crate) fn condition_bits(condition: &ClassicalCondition) -> PyPrismResult<Vec<usize>> {
    Ok(match condition {
        ClassicalCondition::BitIsOne(bit) | ClassicalCondition::BitIsZero(bit) => vec![*bit],
        ClassicalCondition::Parity { bits, .. } => bits.to_vec(),
        ClassicalCondition::RegisterEquals { offset, size, .. }
        | ClassicalCondition::RegisterNotEquals { offset, size, .. } => {
            (*offset..offset + size).collect()
        }
        other => {
            return Err(invalid(format!(
                "condition {other:?} is newer than this binding"
            )));
        }
    })
}

fn check_condition(num_bits: usize, condition: &ClassicalCondition) -> PyPrismResult<()> {
    for bit in condition_bits(condition)? {
        check_classical_bit(num_bits, bit)?;
    }
    Ok(())
}

fn check_register(size: usize, value: u64) -> PyPrismResult<()> {
    if size == 0 || size > 64 {
        return Err(invalid(format!(
            "register conditions span 1 to 64 bits, got {size}"
        )));
    }
    if size < 64 && value >> size != 0 {
        return Err(invalid(format!(
            "value {value} does not fit in a {size}-bit register"
        )));
    }
    Ok(())
}

#[pymethods]
impl PyClassicalCondition {
    /// Holds when classical bit `bit` reads `value`.
    #[staticmethod]
    #[pyo3(signature = (bit, value = true))]
    fn bit(bit: usize, value: bool) -> Self {
        Self(if value {
            ClassicalCondition::BitIsOne(bit)
        } else {
            ClassicalCondition::BitIsZero(bit)
        })
    }

    /// Holds when the XOR of `bits` equals `expected`.
    #[staticmethod]
    #[pyo3(signature = (bits, expected = true))]
    fn parity(bits: Vec<usize>, expected: bool) -> PyPrismResult<Self> {
        if bits.is_empty() {
            return Err(invalid("parity condition needs at least one bit"));
        }
        Ok(Self(ClassicalCondition::Parity {
            bits: bits.into_boxed_slice(),
            expected,
        }))
    }

    /// Holds when bits `offset .. offset + size`, read with `offset` as the
    /// least significant bit, equal `value`.
    #[staticmethod]
    fn register_equals(offset: usize, size: usize, value: u64) -> PyPrismResult<Self> {
        check_register(size, value)?;
        Ok(Self(ClassicalCondition::RegisterEquals {
            offset,
            size,
            value,
        }))
    }

    /// The negation of `register_equals` over the same bits.
    #[staticmethod]
    fn register_not_equals(offset: usize, size: usize, value: u64) -> PyPrismResult<Self> {
        check_register(size, value)?;
        Ok(Self(ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        }))
    }

    /// The condition that holds exactly when this one does not.
    fn negate(&self) -> Self {
        Self(self.0.negate())
    }

    fn __invert__(&self) -> Self {
        self.negate()
    }

    /// Classical bits the condition reads.
    #[getter]
    fn bits(&self) -> PyPrismResult<Vec<usize>> {
        condition_bits(&self.0)
    }

    /// Evaluate against a classical record, bit `i` at index `i`.
    fn evaluate(&self, classical_bits: Vec<bool>) -> PyPrismResult<bool> {
        for bit in condition_bits(&self.0)? {
            check_classical_bit(classical_bits.len(), bit)?;
        }
        Ok(self.0.evaluate(&classical_bits))
    }

    #[staticmethod]
    fn _from_pickle(data: &[u8]) -> PyPrismResult<Self> {
        let mut r = Reader::new(data, Kind::Condition)?;
        let condition = codec::read_condition(&mut r)?;
        r.finish()?;
        Ok(Self(condition))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduced<'py>> {
        let mut w = Writer::new(Kind::Condition);
        codec::write_condition(&mut w, &slf.get().0)?;
        reduce(slf.as_any(), w.finish())
    }

    fn __repr__(&self) -> String {
        format!("ClassicalCondition({:?})", self.0)
    }
}

/// Replay `instructions` onto `builder`, which has no generic append.
fn replay(builder: &mut CircuitBuilder, instructions: Vec<Instruction>) -> PyPrismResult<()> {
    for inst in instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                builder.gate(gate, &targets);
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                builder.measure(qubit, classical_bit);
            }
            Instruction::Reset { qubit } => {
                builder.reset(qubit);
            }
            Instruction::Barrier { qubits } => {
                builder.barrier(&qubits);
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                builder.conditional(condition, gate, &targets);
            }
            Instruction::Region(region) => {
                let body = region.body().to_vec();
                let mut nested = Ok(());
                builder.guarded(region.condition().clone(), |inner| {
                    nested = replay(inner, body);
                });
                nested?;
            }
            Instruction::Save { label, .. } => {
                return Err(invalid(format!(
                    "save point `{label}` cannot sit inside a guarded region"
                )));
            }
        }
    }
    Ok(())
}

/// Call `body` on a fresh builder of the given width and take what it appended.
fn region_body(
    body: &Bound<'_, PyAny>,
    num_qubits: usize,
    num_bits: usize,
) -> PyResult<Vec<Instruction>> {
    let builder = Bound::new(
        body.py(),
        PyCircuitBuilder {
            inner: CircuitBuilder::new_with_classical(num_qubits, num_bits),
        },
    )?;
    body.call1((builder.clone(),))?;
    let builder = builder.borrow();
    let circuit = builder.inner.circuit();
    if circuit.num_qubits != num_qubits || circuit.num_classical_bits != num_bits {
        return Err(invalid(
            "a guarded body cannot widen the circuit (measure_pauli_product or measure_all)",
        )
        .into());
    }
    Ok(circuit.instructions.clone())
}

/// True when a measurement in `instructions` writes one of `bits`.
fn writes_any(instructions: &[Instruction], bits: &[usize]) -> bool {
    instructions.iter().any(|inst| match inst {
        Instruction::Measure { classical_bit, .. } => bits.contains(classical_bit),
        Instruction::Region(region) => writes_any(region.body(), bits),
        _ => false,
    })
}

/// What a save point records.
#[pyclass(name = "SaveSpec", module = "prism_q", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PySaveSpec {
    StateVector,
    Probabilities,
    DensityMatrix,
}

#[pymethods]
impl PySaveSpec {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<ReducedMember<'py>> {
        let name = match *slf.borrow() {
            PySaveSpec::StateVector => "StateVector",
            PySaveSpec::Probabilities => "Probabilities",
            PySaveSpec::DensityMatrix => "DensityMatrix",
        };
        reduce_member(slf.as_any(), name)
    }
}

impl PySaveSpec {
    fn to_core(self) -> SaveSpec {
        match self {
            PySaveSpec::StateVector => SaveSpec::StateVector,
            PySaveSpec::Probabilities => SaveSpec::Probabilities,
            PySaveSpec::DensityMatrix => SaveSpec::DensityMatrix,
        }
    }
}

#[pymethods]
impl PyCircuit {
    #[new]
    #[pyo3(signature = (num_qubits, num_classical_bits = 0))]
    fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self(Circuit::new(num_qubits, num_classical_bits))
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits
    }

    #[getter]
    fn num_classical_bits(&self) -> usize {
        self.0.num_classical_bits
    }

    fn gate_count(&self) -> usize {
        self.0.gate_count()
    }

    fn t_count(&self) -> usize {
        self.0.t_count()
    }

    /// Layers when every operation takes the earliest layer its qubits are free
    /// in. A measurement counts as one layer and a barrier as none.
    fn depth(&self) -> usize {
        self.0.depth()
    }

    fn is_clifford_only(&self) -> bool {
        self.0.is_clifford_only()
    }

    /// Qubit and classical bit of every measurement, as `(qubit, bit)` pairs in
    /// record order.
    ///
    /// Measurements inside a guarded region are included: whether the region is
    /// taken is a runtime fact, so the map covers what the circuit can write.
    fn measurement_map(&self) -> Vec<(usize, usize)> {
        self.0.measurement_map()
    }

    fn add_gate(&mut self, gate: &PyGate, targets: Vec<usize>) -> PyPrismResult<()> {
        check_targets(self.0.num_qubits, gate.inner(), &targets)?;
        self.0.add_gate(gate.inner().clone(), &targets);
        Ok(())
    }

    /// Append the Pauli rotation `exp(-i * theta * P / 2)` for the Pauli string
    /// `P` given as `(qubit, axis)` factors, where `axis` is one of `"X"`,
    /// `"Y"`, `"Z"` and identity factors are omitted.
    ///
    /// A weight-1 string lowers to `rx`, `ry`, or `rz` and a two-qubit `ZZ`
    /// string to `rzz`, so fusion and Clifford recognition keep firing on them;
    /// any other string appends the native multi-qubit rotation.
    fn add_pauli_rotation(
        &mut self,
        theta: f64,
        factors: Vec<(usize, String)>,
    ) -> PyPrismResult<()> {
        let terms = check_pauli_factors(self.0.num_qubits, factors)?;
        self.0.add_pauli_rotation(theta, &terms);
        Ok(())
    }

    fn add_measure(&mut self, qubit: usize, classical_bit: usize) -> PyPrismResult<()> {
        check_qubit(self.0.num_qubits, qubit, "qubit")?;
        check_classical_bit(self.0.num_classical_bits, classical_bit)?;
        self.0.add_measure(qubit, classical_bit);
        Ok(())
    }

    fn add_reset(&mut self, qubit: usize) -> PyPrismResult<()> {
        check_qubit(self.0.num_qubits, qubit, "qubit")?;
        self.0.add_reset(qubit);
        Ok(())
    }

    /// Append a save point recording `spec` under `label`.
    ///
    /// A save observes the whole register and is a fusion barrier across every
    /// qubit. Only `run` returns the records; every other terminal declines a
    /// circuit carrying one.
    fn add_save(&mut self, spec: PySaveSpec, label: String) {
        self.0.add_save(spec.to_core(), label);
    }

    #[getter]
    fn save_count(&self) -> usize {
        self.0.save_count()
    }

    fn add_barrier(&mut self, qubits: Vec<usize>) -> PyPrismResult<()> {
        for (idx, &qubit) in qubits.iter().enumerate() {
            check_qubit(self.0.num_qubits, qubit, format!("qubits[{idx}]"))?;
        }
        self.0.add_barrier(&qubits);
        Ok(())
    }

    /// Render as an OpenQASM 3.0 program that `parse_qasm` reads back.
    ///
    /// A multi-letter Pauli rotation keeps its `r<letters>` spelling, an
    /// extension only PRISM-Q parses; `expand_pauli_rotations=True` lowers it
    /// to basis changes around a CNOT ladder for other toolchains. Save points
    /// and dense gates on three or more qubits have no spelling and raise.
    #[pyo3(signature = (*, expand_pauli_rotations = false))]
    fn to_qasm(&self, expand_pauli_rotations: bool) -> PyPrismResult<String> {
        if expand_pauli_rotations {
            let expanded = prism_q::circuit::expand_pauli_rotations(&self.0);
            Ok(qasm_export::to_qasm3(&expanded)?)
        } else {
            Ok(qasm_export::to_qasm3(&self.0)?)
        }
    }

    /// Text wire diagram, folded at `fold_width` columns. Past 64 qubits or
    /// 500 moments it returns `summary()` instead.
    #[pyo3(signature = (
        *,
        fold_width = TextOptions::default().fold_width,
        show_idle_wires = true,
        show_barriers = true,
        max_qubits = None,
        max_moments = None,
    ))]
    fn draw(
        &self,
        fold_width: usize,
        show_idle_wires: bool,
        show_barriers: bool,
        max_qubits: Option<usize>,
        max_moments: Option<usize>,
    ) -> String {
        self.0.draw(&TextOptions {
            fold_width,
            show_idle_wires,
            show_barriers,
            max_qubits,
            max_moments,
        })
    }

    /// Gate-density heatmap of qubits by moments as text, bucketed to fit
    /// `fold_width` columns.
    #[pyo3(signature = (
        *,
        fold_width = TextOptions::default().fold_width,
        show_idle_wires = true,
        show_barriers = true,
        max_qubits = None,
        max_moments = None,
    ))]
    fn heatmap(
        &self,
        fold_width: usize,
        show_idle_wires: bool,
        show_barriers: bool,
        max_qubits: Option<usize>,
        max_moments: Option<usize>,
    ) -> String {
        self.0.heatmap(&TextOptions {
            fold_width,
            show_idle_wires,
            show_barriers,
            max_qubits,
            max_moments,
        })
    }

    /// Gate counts, connectivity, and depth profile as text.
    fn summary(&self) -> String {
        self.0.summary()
    }

    /// Self-contained SVG wire diagram. Lengths are SVG user units and
    /// `font_size` is in pixels. `ellipsis` as `(first, last)` draws only those
    /// leading and trailing moments when the circuit does not fit.
    #[pyo3(signature = (
        *,
        dark_mode = false,
        auto_theme = false,
        animate = true,
        compact = false,
        show_legend = false,
        show_stats_header = false,
        show_topology = false,
        show_idle_wires = true,
        show_barriers = true,
        max_qubits = None,
        max_moments = None,
        ellipsis = None,
        wire_spacing = SvgOptions::default().wire_spacing,
        moment_width = SvgOptions::default().moment_width,
        gate_height = SvgOptions::default().gate_height,
        gate_min_width = SvgOptions::default().gate_min_width,
        font_size = SvgOptions::default().font_size,
        control_radius = SvgOptions::default().control_radius,
        padding = (
            SvgOptions::default().padding_left,
            SvgOptions::default().padding_top,
            SvgOptions::default().padding_right,
            SvgOptions::default().padding_bottom,
        ),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn to_svg(
        &self,
        dark_mode: bool,
        auto_theme: bool,
        animate: bool,
        compact: bool,
        show_legend: bool,
        show_stats_header: bool,
        show_topology: bool,
        show_idle_wires: bool,
        show_barriers: bool,
        max_qubits: Option<usize>,
        max_moments: Option<usize>,
        ellipsis: Option<(usize, usize)>,
        wire_spacing: f64,
        moment_width: f64,
        gate_height: f64,
        gate_min_width: f64,
        font_size: f64,
        control_radius: f64,
        padding: (f64, f64, f64, f64),
    ) -> String {
        self.0.to_svg(&SvgOptions {
            dark_mode,
            auto_theme,
            animate,
            compact,
            show_legend,
            show_stats_header,
            show_topology,
            show_idle_wires,
            show_barriers,
            max_qubits,
            max_moments,
            ellipsis_mode: ellipsis,
            wire_spacing,
            moment_width,
            gate_height,
            gate_min_width,
            font_size,
            control_radius,
            padding_left: padding.0,
            padding_top: padding.1,
            padding_right: padding.2,
            padding_bottom: padding.3,
        })
    }

    /// Gate-density heatmap as self-contained SVG, with marginal activity bars
    /// and a color legend.
    #[pyo3(signature = (*, dark_mode = false, auto_theme = false))]
    fn to_svg_heatmap(&self, dark_mode: bool, auto_theme: bool) -> String {
        self.0.to_svg_heatmap(&SvgOptions {
            dark_mode,
            auto_theme,
            ..SvgOptions::default()
        })
    }

    /// Jupyter rich display: a static diagram following the page theme, cut to
    /// 64 wires and 200 moments.
    fn _repr_svg_(&self) -> String {
        self.0.to_svg(&SvgOptions {
            auto_theme: true,
            animate: false,
            max_qubits: Some(REPR_MAX_QUBITS),
            max_moments: Some(REPR_MAX_MOMENTS),
            ..SvgOptions::default()
        })
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    #[staticmethod]
    fn _from_pickle(data: &[u8]) -> PyPrismResult<Self> {
        Ok(Self(codec::decode_circuit(data)?))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduced<'py>> {
        let data = codec::encode_circuit(&slf.borrow().0)?;
        reduce(slf.as_any(), data)
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, num_classical_bits={}, gates={})",
            self.0.num_qubits,
            self.0.num_classical_bits,
            self.0.gate_count()
        )
    }
}

impl PyCircuit {
    pub fn inner(&self) -> &Circuit {
        &self.0
    }
}

/// Circuit builder whose gate methods return the builder for chaining;
/// [`build`](Self::build) extracts the [`Circuit`].
#[pyclass(name = "CircuitBuilder", module = "prism_q")]
pub struct PyCircuitBuilder {
    inner: CircuitBuilder,
}

#[pymethods]
impl PyCircuitBuilder {
    #[new]
    #[pyo3(signature = (num_qubits, num_classical_bits = 0))]
    fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self {
            inner: CircuitBuilder::new_with_classical(num_qubits, num_classical_bits),
        }
    }

    fn id(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.id(q);
        Ok(slf)
    }
    fn x(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.x(q);
        Ok(slf)
    }
    fn y(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.y(q);
        Ok(slf)
    }
    fn z(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.z(q);
        Ok(slf)
    }
    fn h(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.h(q);
        Ok(slf)
    }
    fn s(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.s(q);
        Ok(slf)
    }
    fn sdg(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.sdg(q);
        Ok(slf)
    }
    fn t(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.t(q);
        Ok(slf)
    }
    fn tdg(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.tdg(q);
        Ok(slf)
    }
    fn sx(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.sx(q);
        Ok(slf)
    }
    fn sxdg(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.sxdg(q);
        Ok(slf)
    }

    fn rx(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.rx(theta, q);
        Ok(slf)
    }
    fn ry(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.ry(theta, q);
        Ok(slf)
    }
    fn rz(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.rz(theta, q);
        Ok(slf)
    }
    fn p(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.p(theta, q);
        Ok(slf)
    }

    fn rzz(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        slf.inner.rzz(theta, q0, q1);
        Ok(slf)
    }

    /// Append the Pauli rotation `exp(-i * theta * P / 2)`, taking the
    /// `(qubit, axis)` factors `Circuit.add_pauli_rotation` takes and lowering
    /// the same way. Chain `.param(slot)` after it to make the angle bindable.
    fn pauli_rotation(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        factors: Vec<(usize, String)>,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let terms = check_pauli_factors(slf.inner.circuit().num_qubits, factors)?;
        slf.inner.pauli_rotation(theta, &terms);
        Ok(slf)
    }

    /// Bind the most recently appended gate to parameter `slot` for
    /// `Simulation.expectation_gradient` and `PreparedCircuit`. Several gates
    /// may share a slot (their gradients accumulate and binding writes one
    /// angle to each). Example: `builder.rz(theta, q).param(0)`.
    fn param(mut slf: PyRefMut<'_, Self>, slot: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        let bindable = matches!(
            slf.inner.circuit().instructions.last(),
            Some(Instruction::Gate { gate, .. }) if gate.pauli_generator().is_some()
        );
        if !bindable {
            return Err(invalid(
                "param() requires the last appended instruction to be a gate carrying an angle (rx, ry, rz, rzz, p, pauli_rot)",
            ));
        }
        slf.inner.param(slot);
        Ok(slf)
    }

    /// Deprecated alias of `param`, to be removed in the next minor release.
    fn trainable(slf: PyRefMut<'_, Self>, slot: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        Self::param(slf, slot)
    }

    /// The `(instruction_index, parameter_slot)` links recorded by `param`, for
    /// passing to `Simulation.expectation_gradient`.
    fn parameter_links(&self) -> Vec<(usize, usize)> {
        self.inner
            .parameters()
            .links()
            .iter()
            .map(|l| (l.instruction, l.slot))
            .collect()
    }

    /// The parameter set recorded by `param`, for `PreparedCircuit` and
    /// `Parameters.bind`. Pinned to the circuit as it stands, so binding after
    /// further edits fails rather than writing the wrong gates.
    fn parameters(&self) -> PyParameters {
        let circuit = self.inner.circuit();
        PyParameters::against(self.inner.parameters().clone().pinned_to(circuit), circuit)
    }

    fn cx(
        mut slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        slf.inner.cx(control, target);
        Ok(slf)
    }

    fn cz(mut slf: PyRefMut<'_, Self>, q0: usize, q1: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        slf.inner.cz(q0, q1);
        Ok(slf)
    }

    fn swap(
        mut slf: PyRefMut<'_, Self>,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        slf.inner.swap(q0, q1);
        Ok(slf)
    }

    fn cphase(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        slf.inner.cphase(theta, control, target);
        Ok(slf)
    }

    fn cu<'py>(
        mut slf: PyRefMut<'py, Self>,
        matrix: &Bound<'_, PyAny>,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        let mat = crate::gate::extract_2x2(matrix)?;
        slf.inner.cu(mat, control, target);
        Ok(slf)
    }

    fn mcu<'py>(
        mut slf: PyRefMut<'py, Self>,
        matrix: &Bound<'_, PyAny>,
        controls: Vec<usize>,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        check_mcu_targets(slf.inner.circuit().num_qubits, &controls, target)?;
        let mat = crate::gate::extract_2x2(matrix)?;
        slf.inner.mcu(mat, &controls, target);
        Ok(slf)
    }

    /// Append the general rotation `U(theta, phi, lam)` of OpenQASM's `u` and
    /// `u3`, as one fused matrix.
    fn u(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        phi: f64,
        lam: f64,
        q: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.u(theta, phi, lam, q);
        Ok(slf)
    }

    /// Append `exp(-i * theta * XX / 2)`; `.param(slot)` may follow.
    fn rxx(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        check_distinct(&[q0, q1])?;
        slf.inner.rxx(theta, q0, q1);
        Ok(slf)
    }

    /// Append `exp(-i * theta * YY / 2)`; `.param(slot)` may follow.
    fn ryy(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        check_distinct(&[q0, q1])?;
        slf.inner.ryy(theta, q0, q1);
        Ok(slf)
    }

    fn cy(
        mut slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        check_distinct(&[control, target])?;
        slf.inner.cy(control, target);
        Ok(slf)
    }

    fn ch(
        mut slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        check_distinct(&[control, target])?;
        slf.inner.ch(control, target);
        Ok(slf)
    }

    fn crx(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        check_distinct(&[control, target])?;
        slf.inner.crx(theta, control, target);
        Ok(slf)
    }

    fn cry(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        check_distinct(&[control, target])?;
        slf.inner.cry(theta, control, target);
        Ok(slf)
    }

    fn crz(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        check_distinct(&[control, target])?;
        slf.inner.crz(theta, control, target);
        Ok(slf)
    }

    /// Append iSWAP, lowered to the six Clifford gates the QASM parser emits.
    fn iswap(
        mut slf: PyRefMut<'_, Self>,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        check_distinct(&[q0, q1])?;
        slf.inner.iswap(q0, q1);
        Ok(slf)
    }

    /// Append a Toffoli flipping `target` when both controls are |1>.
    fn ccx(
        mut slf: PyRefMut<'_, Self>,
        control0: usize,
        control1: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control0, "control0")?;
        check_qubit(num_qubits, control1, "control1")?;
        check_qubit(num_qubits, target, "target")?;
        check_distinct(&[control0, control1, target])?;
        slf.inner.ccx(control0, control1, target);
        Ok(slf)
    }

    /// Append a controlled swap of `q0` and `q1`.
    fn cswap(
        mut slf: PyRefMut<'_, Self>,
        control: usize,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        check_distinct(&[control, q0, q1])?;
        slf.inner.cswap(control, q0, q1);
        Ok(slf)
    }

    fn measure(
        mut slf: PyRefMut<'_, Self>,
        qubit: usize,
        classical_bit: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let circuit = slf.inner.circuit();
        check_qubit(circuit.num_qubits, qubit, "qubit")?;
        check_classical_bit(circuit.num_classical_bits, classical_bit)?;
        slf.inner.measure(qubit, classical_bit);
        Ok(slf)
    }

    fn measure_all(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.measure_all();
        slf
    }

    /// Measure `qubit` along `axis` (`"X"`, `"Y"` or `"Z"`) into
    /// `classical_bit`, `+1` reading False. The qubit is left in the Z
    /// eigenstate, not the `axis` one.
    fn measure_in_basis<'py>(
        mut slf: PyRefMut<'py, Self>,
        qubit: usize,
        axis: &str,
        classical_bit: usize,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let circuit = slf.inner.circuit();
        check_qubit(circuit.num_qubits, qubit, "qubit")?;
        check_classical_bit(circuit.num_classical_bits, classical_bit)?;
        let axis = parse_axis(axis)?;
        slf.inner.measure_in_basis(qubit, axis, classical_bit);
        Ok(slf)
    }

    /// Measure the Pauli product over `(qubit, axis)` factors into
    /// `classical_bit`, `+1` reading False, leaving the named qubits in the
    /// post-measurement eigenstate.
    ///
    /// The parity collects on one extra qubit appended past the register on the
    /// first call, so the built circuit is one qubit wider. Later calls reset
    /// that qubit, which takes the circuit off the compiled sampling route.
    fn measure_pauli_product(
        mut slf: PyRefMut<'_, Self>,
        factors: Vec<(usize, String)>,
        classical_bit: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let circuit = slf.inner.circuit();
        let terms = check_pauli_factors(circuit.num_qubits, factors)?;
        check_classical_bit(circuit.num_classical_bits, classical_bit)?;
        slf.inner.measure_pauli_product(&terms, classical_bit);
        Ok(slf)
    }

    fn reset(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, qubit, "qubit")?;
        slf.inner.reset(qubit);
        Ok(slf)
    }

    /// Append `gate` on `targets`, applied only when `condition` holds.
    fn conditional<'py>(
        mut slf: PyRefMut<'py, Self>,
        condition: PyClassicalCondition,
        gate: &PyGate,
        targets: Vec<usize>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let circuit = slf.inner.circuit();
        check_targets(circuit.num_qubits, gate.inner(), &targets)?;
        check_condition(circuit.num_classical_bits, &condition.0)?;
        slf.inner
            .conditional(condition.0, gate.inner().clone(), &targets);
        Ok(slf)
    }

    /// Append a region that runs only when `condition` holds, and with
    /// `else_body` one that runs only when it does not.
    ///
    /// Each body is called with a fresh `CircuitBuilder` of the same width,
    /// whose indices are this circuit's; whatever it appends, measurement and
    /// reset included, becomes the region. An `else_body` needs a `body` that
    /// does not measure into a bit the condition reads.
    #[pyo3(signature = (condition, body, else_body = None))]
    fn guarded<'py>(
        slf: Bound<'py, Self>,
        condition: PyClassicalCondition,
        body: Bound<'py, PyAny>,
        else_body: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let (num_qubits, num_bits) = {
            let builder = slf.borrow();
            let circuit = builder.inner.circuit();
            (circuit.num_qubits, circuit.num_classical_bits)
        };
        check_condition(num_bits, &condition.0)?;
        let then_body = region_body(&body, num_qubits, num_bits)?;
        let else_body = else_body
            .map(|body| region_body(&body, num_qubits, num_bits))
            .transpose()?;
        if else_body.is_some() && writes_any(&then_body, &condition_bits(&condition.0)?) {
            return Err(invalid(
                "an else branch needs a body that does not measure into a bit the condition reads",
            )
            .into());
        }
        {
            let mut builder = slf.borrow_mut();
            let mut replayed = Ok(());
            builder.inner.guarded(condition.0.clone(), |inner| {
                replayed = replay(inner, then_body);
            });
            replayed?;
            if let Some(else_body) = else_body {
                let mut replayed = Ok(());
                builder.inner.guarded(condition.0.negate(), |inner| {
                    replayed = replay(inner, else_body);
                });
                replayed?;
            }
        }
        Ok(slf)
    }

    fn barrier(
        mut slf: PyRefMut<'_, Self>,
        qubits: Vec<usize>,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        for (idx, &qubit) in qubits.iter().enumerate() {
            check_qubit(num_qubits, qubit, format!("qubits[{idx}]"))?;
        }
        slf.inner.barrier(&qubits);
        Ok(slf)
    }

    fn gate<'py>(
        mut slf: PyRefMut<'py, Self>,
        gate: &PyGate,
        targets: Vec<usize>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        check_targets(slf.inner.circuit().num_qubits, gate.inner(), &targets)?;
        slf.inner.gate(gate.inner().clone(), &targets);
        Ok(slf)
    }

    /// Return a copy of the circuit so far. The builder stays usable afterward.
    fn build(&self) -> PyCircuit {
        PyCircuit(self.inner.circuit().clone())
    }
}

/// Parse an OpenQASM 3.0 (or 2.0-compatible) string into a [`Circuit`].
#[pyfunction]
pub fn parse_qasm(source: &str) -> PyPrismResult<PyCircuit> {
    Ok(PyCircuit(openqasm::parse(source)?))
}

/// Return a circuit template and its named input slots in declaration order.
#[pyfunction]
pub fn parse_qasm_parametric(source: &str) -> PyPrismResult<(PyCircuit, PyParameters)> {
    let (circuit, parameters) = openqasm::parse_parametric(source)?;
    let parameters = PyParameters::against(parameters, &circuit);
    Ok((PyCircuit(circuit), parameters))
}

macro_rules! circuit_fn {
    ($name:ident, $call:path, ($($arg:ident : $ty:ty),*), ($($sig:tt)*)) => {
        #[pyfunction]
        #[pyo3(signature = ($($sig)*))]
        fn $name($($arg : $ty),*) -> PyCircuit {
            PyCircuit($call($($arg),*))
        }
    };
}

circuit_fn!(qft, circuits::qft_circuit, (n: usize), (n));
circuit_fn!(random, circuits::random_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(hardware_efficient_ansatz, circuits::hardware_efficient_ansatz, (n: usize, layers: usize, seed: u64), (n, layers, seed = 42));
circuit_fn!(clifford_heavy, circuits::clifford_heavy_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(clifford_random_pairs, circuits::clifford_random_pairs, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(qaoa, circuits::qaoa_circuit, (n: usize, layers: usize, seed: u64), (n, layers, seed = 42));
circuit_fn!(single_qubit_rotation, circuits::single_qubit_rotation_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(quantum_volume, circuits::quantum_volume_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(cz_chain, circuits::cz_chain_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(phase_estimation, circuits::phase_estimation_circuit, (n: usize), (n));
circuit_fn!(independent_bell_pairs, circuits::independent_bell_pairs, (n_pairs: usize), (n_pairs));
circuit_fn!(independent_random_blocks, circuits::independent_random_blocks, (num_blocks: usize, block_size: usize, depth: usize, seed: u64), (num_blocks, block_size, depth, seed = 42));
circuit_fn!(local_clifford_blocks, circuits::local_clifford_blocks, (num_blocks: usize, block_size: usize, depth: usize, seed: u64), (num_blocks, block_size, depth, seed = 42));

#[pyfunction]
#[pyo3(signature = (n, depth, t_fraction = 0.1, seed = 42))]
fn clifford_t(n: usize, depth: usize, t_fraction: f64, seed: u64) -> PyCircuit {
    PyCircuit(circuits::clifford_t_circuit(n, depth, t_fraction, seed))
}

#[pyfunction]
fn ghz(n: usize) -> PyPrismResult<PyCircuit> {
    if n == 0 {
        return Err(invalid("ghz requires at least one qubit"));
    }
    Ok(PyCircuit(circuits::ghz_circuit(n)))
}

#[pyfunction]
fn w_state(n: usize) -> PyPrismResult<PyCircuit> {
    if n == 0 {
        return Err(invalid("w_state requires at least one qubit"));
    }
    Ok(PyCircuit(circuits::w_state_circuit(n)))
}

/// Build and register the `prism_q.circuits` submodule.
pub fn register_circuits(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "circuits")?;
    m.add_function(wrap_pyfunction!(qft, &m)?)?;
    m.add_function(wrap_pyfunction!(ghz, &m)?)?;
    m.add_function(wrap_pyfunction!(w_state, &m)?)?;
    m.add_function(wrap_pyfunction!(random, &m)?)?;
    m.add_function(wrap_pyfunction!(hardware_efficient_ansatz, &m)?)?;
    m.add_function(wrap_pyfunction!(clifford_heavy, &m)?)?;
    m.add_function(wrap_pyfunction!(clifford_random_pairs, &m)?)?;
    m.add_function(wrap_pyfunction!(qaoa, &m)?)?;
    m.add_function(wrap_pyfunction!(single_qubit_rotation, &m)?)?;
    m.add_function(wrap_pyfunction!(clifford_t, &m)?)?;
    m.add_function(wrap_pyfunction!(quantum_volume, &m)?)?;
    m.add_function(wrap_pyfunction!(cz_chain, &m)?)?;
    m.add_function(wrap_pyfunction!(phase_estimation, &m)?)?;
    m.add_function(wrap_pyfunction!(independent_bell_pairs, &m)?)?;
    m.add_function(wrap_pyfunction!(independent_random_blocks, &m)?)?;
    m.add_function(wrap_pyfunction!(local_clifford_blocks, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
