//! Versioned binary encoding behind pickling. Every float travels as its IEEE
//! bits, so a decoded value is identical to the encoded one rather than close.
//!
//! Layout: `b"PQ"`, a kind byte, the format version, then the payload with
//! integers as little-endian `u64`. A newer version or another kind is
//! rejected rather than misread.

use num_complex::Complex64;
use prism_q::circuit::GuardedRegion;
use prism_q::{
    Circuit, ClassicalCondition, Gate, Instruction, McuData, NoiseChannel, NoiseEvent, NoiseModel,
    Parameters, PauliAxis, PauliObservable, PauliTerm, QecBasis, QecNoise, QecOp, QecOptions,
    QecPauli, QecProgram, QecRecordRef, ReadoutError, SaveSpec,
};
use smallvec::SmallVec;

use crate::error::{PyPrismResult, invalid};

const MAGIC: &[u8; 2] = b"PQ";
const VERSION: u8 = 1;

#[derive(Clone, Copy)]
#[repr(u8)]
pub(crate) enum Kind {
    Circuit = b'C',
    Gate = b'G',
    NoiseChannel = b'K',
    NoiseModel = b'N',
    Parameters = b'P',
    QecProgram = b'Q',
    PauliObservable = b'O',
    Condition = b'D',
    RecordRef = b'R',
    QecNoise = b'E',
}

pub(crate) struct Writer(Vec<u8>);

impl Writer {
    pub(crate) fn new(kind: Kind) -> Self {
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(MAGIC);
        out.push(kind as u8);
        out.push(VERSION);
        Self(out)
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.0
    }

    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn c64(&mut self, value: Complex64) {
        self.f64(value.re);
        self.f64(value.im);
    }

    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.0.extend_from_slice(value.as_bytes());
    }

    fn usizes(&mut self, values: &[usize]) {
        self.usize(values.len());
        for &value in values {
            self.usize(value);
        }
    }

    fn f64s(&mut self, values: &[f64]) {
        self.usize(values.len());
        for &value in values {
            self.f64(value);
        }
    }

    fn c64s(&mut self, values: impl ExactSizeIterator<Item = Complex64>) {
        self.usize(values.len());
        for value in values {
            self.c64(value);
        }
    }
}

pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8], kind: Kind) -> PyPrismResult<Self> {
        if data.len() < 4 || &data[..2] != MAGIC {
            return Err(invalid("not a PRISM-Q pickle payload"));
        }
        if data[2] != kind as u8 {
            return Err(invalid(format!(
                "pickle payload holds kind {:?}, expected {:?}",
                data[2] as char, kind as u8 as char
            )));
        }
        if data[3] != VERSION {
            return Err(invalid(format!(
                "pickle payload has format version {}, this build reads version {VERSION}",
                data[3]
            )));
        }
        Ok(Self { data, pos: 4 })
    }

    pub(crate) fn finish(self) -> PyPrismResult<()> {
        if self.pos != self.data.len() {
            return Err(invalid("pickle payload has trailing bytes"));
        }
        Ok(())
    }

    fn take(&mut self, len: usize) -> PyPrismResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| invalid("pickle payload is truncated"))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> PyPrismResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> PyPrismResult<u64> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("took eight bytes"),
        ))
    }

    fn usize(&mut self) -> PyPrismResult<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid("pickle payload index overflows usize"))
    }

    /// A length prefix, bounded by the bytes left so a corrupt prefix cannot
    /// drive a huge allocation.
    fn len(&mut self, min_item_bytes: usize) -> PyPrismResult<usize> {
        let len = self.usize()?;
        if len.saturating_mul(min_item_bytes.max(1)) > self.data.len() - self.pos {
            return Err(invalid("pickle payload is truncated"));
        }
        Ok(len)
    }

    fn bool(&mut self) -> PyPrismResult<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(invalid(format!("pickle payload has bool byte {other}"))),
        }
    }

    fn f64(&mut self) -> PyPrismResult<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn c64(&mut self) -> PyPrismResult<Complex64> {
        Ok(Complex64::new(self.f64()?, self.f64()?))
    }

    fn str(&mut self) -> PyPrismResult<String> {
        let len = self.len(1)?;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| invalid("pickle payload holds a string that is not UTF-8"))
    }

    fn usizes(&mut self) -> PyPrismResult<Vec<usize>> {
        let len = self.len(8)?;
        (0..len).map(|_| self.usize()).collect()
    }

    fn f64s(&mut self) -> PyPrismResult<Vec<f64>> {
        let len = self.len(8)?;
        (0..len).map(|_| self.f64()).collect()
    }

    fn c64s(&mut self) -> PyPrismResult<Vec<Complex64>> {
        let len = self.len(16)?;
        (0..len).map(|_| self.c64()).collect()
    }

    fn matrix<const N: usize>(&mut self) -> PyPrismResult<[[Complex64; N]; N]> {
        let flat = self.c64s()?;
        if flat.len() != N * N {
            return Err(invalid(format!(
                "pickle payload holds {} matrix entries, expected {}",
                flat.len(),
                N * N
            )));
        }
        let mut mat = [[Complex64::new(0.0, 0.0); N]; N];
        for (row, chunk) in mat.iter_mut().zip(flat.chunks_exact(N)) {
            row.copy_from_slice(chunk);
        }
        Ok(mat)
    }
}

fn bad_tag(what: &str, tag: u8) -> crate::error::PyPrismError {
    invalid(format!("pickle payload has unknown {what} tag {tag}"))
}

fn flat<const N: usize>(mat: &[[Complex64; N]; N]) -> impl ExactSizeIterator<Item = Complex64> {
    mat.as_flattened().to_vec().into_iter()
}

fn axis_tag(axis: PauliAxis) -> u8 {
    match axis {
        PauliAxis::X => 0,
        PauliAxis::Y => 1,
        PauliAxis::Z => 2,
    }
}

fn axis_of(tag: u8) -> PyPrismResult<PauliAxis> {
    match tag {
        0 => Ok(PauliAxis::X),
        1 => Ok(PauliAxis::Y),
        2 => Ok(PauliAxis::Z),
        other => Err(bad_tag("Pauli axis", other)),
    }
}

pub(crate) fn write_gate(w: &mut Writer, gate: &Gate) -> PyPrismResult<()> {
    let simple = match gate {
        Gate::Id => Some(0),
        Gate::X => Some(1),
        Gate::Y => Some(2),
        Gate::Z => Some(3),
        Gate::H => Some(4),
        Gate::S => Some(5),
        Gate::Sdg => Some(6),
        Gate::T => Some(7),
        Gate::Tdg => Some(8),
        Gate::SX => Some(9),
        Gate::SXdg => Some(10),
        Gate::Cx => Some(16),
        Gate::Cz => Some(17),
        Gate::Swap => Some(18),
        _ => None,
    };
    if let Some(tag) = simple {
        w.u8(tag);
        return Ok(());
    }
    match gate {
        Gate::Rx(theta) | Gate::Ry(theta) | Gate::Rz(theta) | Gate::P(theta) | Gate::Rzz(theta) => {
            w.u8(match gate {
                Gate::Rx(_) => 11,
                Gate::Ry(_) => 12,
                Gate::Rz(_) => 13,
                Gate::P(_) => 14,
                _ => 15,
            });
            w.f64(*theta);
        }
        Gate::Cu(mat) => {
            w.u8(19);
            w.c64s(flat(mat));
        }
        Gate::Mcu(data) => {
            w.u8(20);
            w.c64s(flat(data.mat()));
            w.u8(data.num_controls());
        }
        Gate::Fused(mat) => {
            w.u8(21);
            w.c64s(flat(mat));
        }
        Gate::Fused2q(mat) => {
            w.u8(22);
            w.c64s(flat(mat));
        }
        Gate::QftBlock { start, num } => {
            w.u8(23);
            w.u8(*start);
            w.u8(*num);
        }
        Gate::PauliRot(data) => {
            w.u8(24);
            w.f64(data.theta());
            w.usize(data.axes().len());
            for &axis in data.axes() {
                w.u8(axis_tag(axis));
            }
        }
        Gate::Unitary(data) => {
            w.u8(25);
            w.usize(data.num_qubits());
            w.c64s(data.matrix().iter().copied());
        }
        other => {
            return Err(invalid(format!(
                "gate `{}` is a fused form with no pickle encoding; pickle the unfused circuit",
                other.name()
            )));
        }
    }
    Ok(())
}

/// Decode a gate. A Pauli rotation needs its targets, so it is returned
/// through the lowering that built it, which fixes the target order.
fn read_gate(r: &mut Reader<'_>, targets: &[usize]) -> PyPrismResult<Gate> {
    let tag = r.u8()?;
    Ok(match tag {
        0 => Gate::Id,
        1 => Gate::X,
        2 => Gate::Y,
        3 => Gate::Z,
        4 => Gate::H,
        5 => Gate::S,
        6 => Gate::Sdg,
        7 => Gate::T,
        8 => Gate::Tdg,
        9 => Gate::SX,
        10 => Gate::SXdg,
        11 => Gate::Rx(r.f64()?),
        12 => Gate::Ry(r.f64()?),
        13 => Gate::Rz(r.f64()?),
        14 => Gate::P(r.f64()?),
        15 => Gate::Rzz(r.f64()?),
        16 => Gate::Cx,
        17 => Gate::Cz,
        18 => Gate::Swap,
        19 => Gate::Cu(Box::new(r.matrix::<2>()?)),
        20 => {
            let mat = r.matrix::<2>()?;
            let num_controls = r.u8()?;
            if num_controls < 2 {
                return Err(invalid(
                    "pickle payload holds an mcu with fewer than 2 controls",
                ));
            }
            Gate::Mcu(Box::new(McuData::new(mat, num_controls)))
        }
        21 => Gate::Fused(Box::new(r.matrix::<2>()?)),
        22 => Gate::Fused2q(Box::new(r.matrix::<4>()?)),
        23 => Gate::QftBlock {
            start: r.u8()?,
            num: r.u8()?,
        },
        24 => {
            let theta = r.f64()?;
            let len = r.len(1)?;
            let axes: Vec<PauliAxis> = (0..len)
                .map(|_| axis_of(r.u8()?))
                .collect::<PyPrismResult<_>>()?;
            if axes.len() != targets.len() || axes.len() < 2 {
                return Err(invalid("pickle payload holds a malformed Pauli rotation"));
            }
            let terms: Vec<PauliTerm> = targets
                .iter()
                .zip(&axes)
                .map(|(&qubit, &axis)| PauliTerm::new(qubit, axis))
                .collect();
            let width = targets.iter().max().map_or(0, |&q| q + 1);
            let mut scratch = Circuit::new(width, 0);
            scratch.add_pauli_rotation(theta, &terms);
            match scratch.instructions.pop() {
                Some(Instruction::Gate {
                    gate,
                    targets: lowered,
                }) if lowered.as_slice() == targets => gate,
                _ => {
                    return Err(invalid(
                        "pickle payload holds a Pauli rotation that does not lower to itself",
                    ));
                }
            }
        }
        25 => {
            let num_qubits = r.usize()?;
            Gate::unitary(r.c64s()?, num_qubits)?
        }
        other => return Err(bad_tag("gate", other)),
    })
}

/// A gate with no targets of its own, as `Gate` pickles; a Pauli rotation
/// never reaches here because Python cannot hold one outside a circuit.
pub(crate) fn read_lone_gate(r: &mut Reader<'_>) -> PyPrismResult<Gate> {
    read_gate(r, &[])
}

pub(crate) fn write_condition(w: &mut Writer, condition: &ClassicalCondition) -> PyPrismResult<()> {
    match condition {
        ClassicalCondition::BitIsOne(bit) => {
            w.u8(0);
            w.usize(*bit);
        }
        ClassicalCondition::BitIsZero(bit) => {
            w.u8(1);
            w.usize(*bit);
        }
        ClassicalCondition::Parity { bits, expected } => {
            w.u8(2);
            w.usizes(bits);
            w.bool(*expected);
        }
        ClassicalCondition::RegisterEquals {
            offset,
            size,
            value,
        } => {
            w.u8(3);
            w.usize(*offset);
            w.usize(*size);
            w.u64(*value);
        }
        ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        } => {
            w.u8(4);
            w.usize(*offset);
            w.usize(*size);
            w.u64(*value);
        }
        other => {
            return Err(invalid(format!(
                "condition {other:?} has no pickle encoding"
            )));
        }
    }
    Ok(())
}

pub(crate) fn read_condition(r: &mut Reader<'_>) -> PyPrismResult<ClassicalCondition> {
    Ok(match r.u8()? {
        0 => ClassicalCondition::BitIsOne(r.usize()?),
        1 => ClassicalCondition::BitIsZero(r.usize()?),
        2 => ClassicalCondition::Parity {
            bits: r.usizes()?.into_boxed_slice(),
            expected: r.bool()?,
        },
        tag @ (3 | 4) => {
            let (offset, size, value) = (r.usize()?, r.usize()?, r.u64()?);
            if tag == 3 {
                ClassicalCondition::RegisterEquals {
                    offset,
                    size,
                    value,
                }
            } else {
                ClassicalCondition::RegisterNotEquals {
                    offset,
                    size,
                    value,
                }
            }
        }
        other => return Err(bad_tag("condition", other)),
    })
}

fn write_instructions(w: &mut Writer, instructions: &[Instruction]) -> PyPrismResult<()> {
    w.usize(instructions.len());
    for inst in instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                w.u8(0);
                w.usizes(targets);
                write_gate(w, gate)?;
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                w.u8(1);
                w.usize(*qubit);
                w.usize(*classical_bit);
            }
            Instruction::Reset { qubit } => {
                w.u8(2);
                w.usize(*qubit);
            }
            Instruction::Barrier { qubits } => {
                w.u8(3);
                w.usizes(qubits);
            }
            Instruction::Save {
                spec,
                label,
                qubits,
            } => {
                w.u8(4);
                w.u8(match spec {
                    SaveSpec::StateVector => 0,
                    SaveSpec::Probabilities => 1,
                    SaveSpec::DensityMatrix => 2,
                    other => {
                        return Err(invalid(format!(
                            "save spec {other:?} has no pickle encoding"
                        )));
                    }
                });
                w.str(label);
                w.usizes(qubits);
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                w.u8(5);
                write_condition(w, condition)?;
                w.usizes(targets);
                write_gate(w, gate)?;
            }
            Instruction::Region(region) => {
                w.u8(6);
                write_condition(w, region.condition())?;
                write_instructions(w, region.body())?;
            }
        }
    }
    Ok(())
}

fn read_instructions(r: &mut Reader<'_>) -> PyPrismResult<Vec<Instruction>> {
    let len = r.len(1)?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(match r.u8()? {
            0 => {
                let targets = r.usizes()?;
                let gate = read_gate(r, &targets)?;
                Instruction::Gate {
                    gate,
                    targets: SmallVec::from_vec(targets),
                }
            }
            1 => Instruction::Measure {
                qubit: r.usize()?,
                classical_bit: r.usize()?,
            },
            2 => Instruction::Reset { qubit: r.usize()? },
            3 => Instruction::Barrier {
                qubits: SmallVec::from_vec(r.usizes()?),
            },
            4 => {
                let spec = match r.u8()? {
                    0 => SaveSpec::StateVector,
                    1 => SaveSpec::Probabilities,
                    2 => SaveSpec::DensityMatrix,
                    other => return Err(bad_tag("save spec", other)),
                };
                Instruction::Save {
                    spec,
                    label: r.str()?,
                    qubits: SmallVec::from_vec(r.usizes()?),
                }
            }
            5 => {
                let condition = read_condition(r)?;
                let targets = r.usizes()?;
                let gate = read_gate(r, &targets)?;
                Instruction::Conditional {
                    condition,
                    gate,
                    targets: SmallVec::from_vec(targets),
                }
            }
            6 => {
                let condition = read_condition(r)?;
                let body = read_instructions(r)?;
                Instruction::Region(Box::new(GuardedRegion::new(condition, body)))
            }
            other => return Err(bad_tag("instruction", other)),
        });
    }
    Ok(out)
}

/// Check every index an instruction names against the register, so a corrupt
/// payload raises here rather than panicking inside a kernel.
fn check_instructions(circuit: &Circuit, instructions: &[Instruction]) -> PyPrismResult<()> {
    let qubit = |q: usize| {
        if q < circuit.num_qubits {
            Ok(())
        } else {
            Err(invalid(format!("pickle payload names qubit {q}")))
        }
    };
    let bit = |b: usize| {
        if b < circuit.num_classical_bits {
            Ok(())
        } else {
            Err(invalid(format!("pickle payload names classical bit {b}")))
        }
    };
    let condition = |c: &ClassicalCondition| -> PyPrismResult<()> {
        crate::circuit::condition_bits(c)?
            .into_iter()
            .try_for_each(bit)
    };
    let gate = |gate: &Gate, targets: &[usize]| -> PyPrismResult<()> {
        if gate.num_qubits() != targets.len() {
            return Err(invalid("pickle payload holds a gate of the wrong arity"));
        }
        targets.iter().try_for_each(|&q| qubit(q))
    };
    for inst in instructions {
        match inst {
            Instruction::Gate { gate: g, targets } => gate(g, targets)?,
            Instruction::Measure {
                qubit: q,
                classical_bit,
            } => {
                qubit(*q)?;
                bit(*classical_bit)?;
            }
            Instruction::Reset { qubit: q } => qubit(*q)?,
            Instruction::Barrier { qubits } | Instruction::Save { qubits, .. } => {
                qubits.iter().try_for_each(|&q| qubit(q))?;
            }
            Instruction::Conditional {
                condition: c,
                gate: g,
                targets,
            } => {
                condition(c)?;
                gate(g, targets)?;
            }
            Instruction::Region(region) => {
                condition(region.condition())?;
                check_instructions(circuit, region.body())?;
            }
        }
    }
    Ok(())
}

pub(crate) fn encode_circuit(circuit: &Circuit) -> PyPrismResult<Vec<u8>> {
    let mut w = Writer::new(Kind::Circuit);
    w.usize(circuit.num_qubits);
    w.usize(circuit.num_classical_bits);
    write_instructions(&mut w, &circuit.instructions)?;
    Ok(w.finish())
}

pub(crate) fn decode_circuit(data: &[u8]) -> PyPrismResult<Circuit> {
    let mut r = Reader::new(data, Kind::Circuit)?;
    let mut circuit = Circuit::new(r.usize()?, r.usize()?);
    circuit.instructions = read_instructions(&mut r)?;
    r.finish()?;
    check_instructions(&circuit, &circuit.instructions)?;
    Ok(circuit)
}

pub(crate) fn write_channel(w: &mut Writer, channel: &NoiseChannel) -> PyPrismResult<()> {
    match channel {
        NoiseChannel::Pauli { px, py, pz } => {
            w.u8(0);
            w.f64s(&[*px, *py, *pz]);
        }
        NoiseChannel::Depolarizing { p } => {
            w.u8(1);
            w.f64(*p);
        }
        NoiseChannel::AmplitudeDamping { gamma } => {
            w.u8(2);
            w.f64(*gamma);
        }
        NoiseChannel::PhaseDamping { gamma } => {
            w.u8(3);
            w.f64(*gamma);
        }
        NoiseChannel::ThermalRelaxation {
            t1,
            t2,
            gate_time,
            excited_population,
        } => {
            w.u8(4);
            w.f64s(&[*t1, *t2, *gate_time, *excited_population]);
        }
        NoiseChannel::TwoQubitDepolarizing { p } => {
            w.u8(5);
            w.f64(*p);
        }
        NoiseChannel::Custom { kraus } => {
            w.u8(6);
            w.usize(kraus.len());
            for mat in kraus {
                w.c64s(flat(mat));
            }
        }
        NoiseChannel::Kraus2q { kraus } => {
            w.u8(7);
            w.usize(kraus.len());
            for mat in kraus {
                w.c64s(flat(mat));
            }
        }
        other => {
            return Err(invalid(format!(
                "noise channel {other:?} has no pickle encoding"
            )));
        }
    }
    Ok(())
}

pub(crate) fn read_channel(r: &mut Reader<'_>) -> PyPrismResult<NoiseChannel> {
    let tag = r.u8()?;
    let fixed = |r: &mut Reader<'_>, n: usize| -> PyPrismResult<Vec<f64>> {
        let values = r.f64s()?;
        if values.len() != n {
            return Err(invalid("pickle payload holds a malformed noise channel"));
        }
        Ok(values)
    };
    Ok(match tag {
        0 => {
            let v = fixed(r, 3)?;
            NoiseChannel::Pauli {
                px: v[0],
                py: v[1],
                pz: v[2],
            }
        }
        1 => NoiseChannel::Depolarizing { p: r.f64()? },
        2 => NoiseChannel::AmplitudeDamping { gamma: r.f64()? },
        3 => NoiseChannel::PhaseDamping { gamma: r.f64()? },
        4 => {
            let v = fixed(r, 4)?;
            NoiseChannel::ThermalRelaxation {
                t1: v[0],
                t2: v[1],
                gate_time: v[2],
                excited_population: v[3],
            }
        }
        5 => NoiseChannel::TwoQubitDepolarizing { p: r.f64()? },
        6 => {
            let len = r.len(64)?;
            NoiseChannel::Custom {
                kraus: (0..len)
                    .map(|_| r.matrix::<2>())
                    .collect::<PyPrismResult<_>>()?,
            }
        }
        7 => {
            let len = r.len(256)?;
            NoiseChannel::Kraus2q {
                kraus: (0..len)
                    .map(|_| r.matrix::<4>())
                    .collect::<PyPrismResult<_>>()?,
            }
        }
        other => return Err(bad_tag("noise channel", other)),
    })
}

pub(crate) fn encode_noise_model(model: &NoiseModel) -> PyPrismResult<Vec<u8>> {
    let mut w = Writer::new(Kind::NoiseModel);
    w.usize(model.after_gate.len());
    for slot in &model.after_gate {
        w.usize(slot.len());
        for event in slot {
            write_channel(&mut w, &event.channel)?;
            w.usizes(&event.qubits);
        }
    }
    w.usize(model.readout.len());
    for entry in &model.readout {
        match entry {
            Some(readout) => {
                w.bool(true);
                w.f64(readout.p01);
                w.f64(readout.p10);
            }
            None => w.bool(false),
        }
    }
    Ok(w.finish())
}

pub(crate) fn decode_noise_model(data: &[u8]) -> PyPrismResult<NoiseModel> {
    let mut r = Reader::new(data, Kind::NoiseModel)?;
    let slots = r.len(8)?;
    let mut after_gate = Vec::with_capacity(slots);
    for _ in 0..slots {
        let events = r.len(1)?;
        let mut slot = Vec::with_capacity(events);
        for _ in 0..events {
            let channel = read_channel(&mut r)?;
            let qubits = SmallVec::from_vec(r.usizes()?);
            slot.push(NoiseEvent { channel, qubits });
        }
        after_gate.push(slot);
    }
    let bits = r.len(1)?;
    let mut readout = Vec::with_capacity(bits);
    for _ in 0..bits {
        readout.push(if r.bool()? {
            Some(ReadoutError {
                p01: r.f64()?,
                p10: r.f64()?,
            })
        } else {
            None
        });
    }
    r.finish()?;
    Ok(NoiseModel {
        after_gate,
        readout,
    })
}

/// The gates a parameter set was pinned against, one per pinned link, so the
/// edit guard survives a round trip; `Parameters` keeps the pin private.
pub(crate) type Pin = Vec<(Gate, SmallVec<[usize; 4]>)>;

pub(crate) fn encode_parameters(params: &Parameters, pin: Option<&Pin>) -> PyPrismResult<Vec<u8>> {
    let mut w = Writer::new(Kind::Parameters);
    w.usize(params.num_slots());
    w.usize(params.links().len());
    for link in params.links() {
        w.usize(link.instruction);
        w.usize(link.slot);
    }
    let named = params.num_slots() > 0 && params.name_of(0).is_some();
    w.bool(named);
    if named {
        for slot in 0..params.num_slots() {
            w.str(params.name_of(slot).unwrap_or_default());
        }
    }
    match pin {
        Some(pin) => {
            w.bool(true);
            w.usize(pin.len());
            for (gate, targets) in pin {
                w.usizes(targets);
                write_gate(&mut w, gate)?;
            }
        }
        None => w.bool(false),
    }
    Ok(w.finish())
}

pub(crate) fn decode_parameters(data: &[u8]) -> PyPrismResult<(Parameters, Option<Pin>)> {
    let mut r = Reader::new(data, Kind::Parameters)?;
    let num_slots = r.usize()?;
    let num_links = r.len(16)?;
    let mut links = Vec::with_capacity(num_links);
    for _ in 0..num_links {
        let (instruction, slot) = (r.usize()?, r.usize()?);
        if slot >= num_slots {
            return Err(invalid(
                "pickle payload links a slot past the declared count",
            ));
        }
        links.push(prism_q::ParamLink { instruction, slot });
    }
    let names = if r.bool()? {
        Some(
            (0..num_slots)
                .map(|_| r.str())
                .collect::<PyPrismResult<Vec<_>>>()?,
        )
    } else {
        None
    };
    let pin = if r.bool()? {
        let len = r.len(9)?;
        let mut pin = Vec::with_capacity(len);
        for _ in 0..len {
            let targets = r.usizes()?;
            let gate = read_gate(&mut r, &targets)?;
            pin.push((gate, SmallVec::from_vec(targets)));
        }
        Some(pin)
    } else {
        None
    };
    r.finish()?;
    let params = restore_parameters(num_slots, links, names, pin.as_ref())?;
    Ok((params, pin))
}

/// Rebuild a parameter set, re-pinning the leading links against a scratch
/// circuit that holds the recorded gates at their instruction indices.
pub(crate) fn restore_parameters(
    num_slots: usize,
    links: Vec<prism_q::ParamLink>,
    names: Option<Vec<String>>,
    pin: Option<&Pin>,
) -> PyPrismResult<Parameters> {
    let pinned = pin.map_or(0, Vec::len);
    if pinned > links.len() {
        return Err(invalid("pickle payload pins more links than it holds"));
    }
    let mut params = Parameters::from_links(links[..pinned].to_vec(), num_slots);
    if let Some(names) = names {
        params = params.with_names(names);
    }
    if let Some(pin) = pin {
        let len = links[..pinned]
            .iter()
            .map(|link| link.instruction + 1)
            .max()
            .unwrap_or(0);
        let width = pin
            .iter()
            .flat_map(|(_, targets)| targets.iter().map(|&q| q + 1))
            .max()
            .unwrap_or(0);
        let mut scratch = Circuit::new(width, 0);
        scratch.instructions = vec![
            Instruction::Barrier {
                qubits: SmallVec::new()
            };
            len
        ];
        for (link, (gate, targets)) in links.iter().zip(pin) {
            scratch.instructions[link.instruction] = Instruction::Gate {
                gate: gate.clone(),
                targets: targets.clone(),
            };
        }
        params = params.pinned_to(&scratch);
    }
    for link in &links[pinned..] {
        params.link(link.instruction, link.slot);
    }
    Ok(params)
}

fn write_records(w: &mut Writer, records: &[QecRecordRef]) -> PyPrismResult<()> {
    w.usize(records.len());
    for record in records {
        write_record(w, record)?;
    }
    Ok(())
}

pub(crate) fn write_record(w: &mut Writer, record: &QecRecordRef) -> PyPrismResult<()> {
    match record {
        QecRecordRef::Absolute(index) => {
            w.u8(0);
            w.usize(*index);
        }
        QecRecordRef::Lookback(distance) => {
            w.u8(1);
            w.usize(*distance);
        }
        other => {
            return Err(invalid(format!(
                "record reference {other:?} has no pickle encoding"
            )));
        }
    }
    Ok(())
}

pub(crate) fn read_record(r: &mut Reader<'_>) -> PyPrismResult<QecRecordRef> {
    Ok(match r.u8()? {
        0 => QecRecordRef::Absolute(r.usize()?),
        1 => QecRecordRef::lookback(r.usize()?)?,
        other => return Err(bad_tag("record reference", other)),
    })
}

fn read_records(r: &mut Reader<'_>) -> PyPrismResult<Vec<QecRecordRef>> {
    let len = r.len(9)?;
    (0..len).map(|_| read_record(r)).collect()
}

fn basis_tag(basis: QecBasis) -> u8 {
    match basis {
        QecBasis::X => 0,
        QecBasis::Y => 1,
        QecBasis::Z => 2,
    }
}

fn basis_of(tag: u8) -> PyPrismResult<QecBasis> {
    match tag {
        0 => Ok(QecBasis::X),
        1 => Ok(QecBasis::Y),
        2 => Ok(QecBasis::Z),
        other => Err(bad_tag("basis", other)),
    }
}

fn write_paulis(w: &mut Writer, terms: &[QecPauli]) {
    w.usize(terms.len());
    for term in terms {
        w.u8(basis_tag(term.basis));
        w.usize(term.qubit);
    }
}

fn read_paulis(r: &mut Reader<'_>) -> PyPrismResult<Vec<QecPauli>> {
    let len = r.len(9)?;
    (0..len)
        .map(|_| Ok(QecPauli::new(basis_of(r.u8()?)?, r.usize()?)))
        .collect()
}

pub(crate) fn write_qec_noise(w: &mut Writer, channel: QecNoise) -> PyPrismResult<()> {
    w.u8(match channel {
        QecNoise::XError(_) => 0,
        QecNoise::ZError(_) => 1,
        QecNoise::Depolarize1(_) => 2,
        QecNoise::Depolarize2(_) => 3,
        other => {
            return Err(invalid(format!(
                "QEC noise {other:?} has no pickle encoding"
            )));
        }
    });
    w.f64(channel.probability());
    Ok(())
}

pub(crate) fn read_qec_noise(r: &mut Reader<'_>) -> PyPrismResult<QecNoise> {
    let tag = r.u8()?;
    let p = r.f64()?;
    Ok(match tag {
        0 => QecNoise::XError(p),
        1 => QecNoise::ZError(p),
        2 => QecNoise::Depolarize1(p),
        3 => QecNoise::Depolarize2(p),
        other => return Err(bad_tag("QEC noise", other)),
    })
}

fn write_ops(w: &mut Writer, ops: &[QecOp]) -> PyPrismResult<()> {
    w.usize(ops.len());
    for op in ops {
        match op {
            QecOp::Gate { gate, targets } => {
                w.u8(0);
                w.usizes(targets);
                write_gate(w, gate)?;
            }
            QecOp::Measure { basis, qubit } => {
                w.u8(1);
                w.u8(basis_tag(*basis));
                w.usize(*qubit);
            }
            QecOp::MeasurePauliProduct { terms } => {
                w.u8(2);
                write_paulis(w, terms);
            }
            QecOp::Reset { basis, qubit } => {
                w.u8(3);
                w.u8(basis_tag(*basis));
                w.usize(*qubit);
            }
            QecOp::Detector { records, coords } => {
                w.u8(4);
                write_records(w, records)?;
                w.f64s(coords);
            }
            QecOp::ObservableInclude {
                observable,
                records,
            } => {
                w.u8(5);
                w.usize(*observable);
                write_records(w, records)?;
            }
            QecOp::ExpectationValue { terms, coefficient } => {
                w.u8(6);
                write_paulis(w, terms);
                w.f64(*coefficient);
            }
            QecOp::Postselect { records, expected } => {
                w.u8(7);
                write_records(w, records)?;
                w.bool(*expected);
            }
            QecOp::Feedforward {
                records,
                expected,
                body,
            } => {
                w.u8(8);
                write_records(w, records)?;
                w.bool(*expected);
                write_ops(w, body)?;
            }
            QecOp::Noise { channel, targets } => {
                w.u8(9);
                write_qec_noise(w, *channel)?;
                w.usizes(targets);
            }
            QecOp::Tick => w.u8(10),
            other => {
                return Err(invalid(format!("QEC op {other:?} has no pickle encoding")));
            }
        }
    }
    Ok(())
}

fn read_ops(r: &mut Reader<'_>) -> PyPrismResult<Vec<QecOp>> {
    let len = r.len(1)?;
    let mut ops = Vec::with_capacity(len);
    for _ in 0..len {
        ops.push(match r.u8()? {
            0 => {
                let targets = r.usizes()?;
                let gate = read_gate(r, &targets)?;
                QecOp::Gate { gate, targets }
            }
            1 => QecOp::Measure {
                basis: basis_of(r.u8()?)?,
                qubit: r.usize()?,
            },
            2 => QecOp::MeasurePauliProduct {
                terms: read_paulis(r)?,
            },
            3 => QecOp::Reset {
                basis: basis_of(r.u8()?)?,
                qubit: r.usize()?,
            },
            4 => QecOp::Detector {
                records: read_records(r)?,
                coords: r.f64s()?,
            },
            5 => QecOp::ObservableInclude {
                observable: r.usize()?,
                records: read_records(r)?,
            },
            6 => QecOp::ExpectationValue {
                terms: read_paulis(r)?,
                coefficient: r.f64()?,
            },
            7 => QecOp::Postselect {
                records: read_records(r)?,
                expected: r.bool()?,
            },
            8 => QecOp::Feedforward {
                records: read_records(r)?,
                expected: r.bool()?,
                body: read_ops(r)?,
            },
            9 => QecOp::Noise {
                channel: read_qec_noise(r)?,
                targets: r.usizes()?,
            },
            10 => QecOp::Tick,
            other => return Err(bad_tag("QEC op", other)),
        });
    }
    Ok(ops)
}

pub(crate) fn encode_qec_program(program: &QecProgram) -> PyPrismResult<Vec<u8>> {
    let mut w = Writer::new(Kind::QecProgram);
    w.usize(program.num_qubits());
    let options = program.options();
    w.usize(options.shots);
    w.u64(options.seed);
    match options.chunk_size {
        Some(size) => {
            w.bool(true);
            w.usize(size);
        }
        None => w.bool(false),
    }
    w.bool(options.keep_measurements);
    write_ops(&mut w, program.ops())?;
    Ok(w.finish())
}

pub(crate) fn decode_qec_program(data: &[u8]) -> PyPrismResult<QecProgram> {
    let mut r = Reader::new(data, Kind::QecProgram)?;
    let num_qubits = r.usize()?;
    let shots = r.usize()?;
    let seed = r.u64()?;
    let chunk_size = if r.bool()? { Some(r.usize()?) } else { None };
    let keep_measurements = r.bool()?;
    let ops = read_ops(&mut r)?;
    r.finish()?;
    let options = QecOptions {
        shots,
        seed,
        chunk_size,
        keep_measurements,
    };
    Ok(QecProgram::from_ops(num_qubits, options, ops)?)
}

pub(crate) fn encode_observable(observable: &PauliObservable) -> Vec<u8> {
    let mut w = Writer::new(Kind::PauliObservable);
    w.usize(observable.terms().len());
    for (coefficient, factors) in observable.terms() {
        w.f64(*coefficient);
        w.usize(factors.len());
        for term in factors {
            w.usize(term.qubit);
            w.u8(axis_tag(term.axis));
        }
    }
    w.finish()
}

pub(crate) fn decode_observable(data: &[u8]) -> PyPrismResult<PauliObservable> {
    let mut r = Reader::new(data, Kind::PauliObservable)?;
    let len = r.len(16)?;
    let mut terms = Vec::with_capacity(len);
    for _ in 0..len {
        let coefficient = r.f64()?;
        let factors = r.len(9)?;
        let factors = (0..factors)
            .map(|_| Ok(PauliTerm::new(r.usize()?, axis_of(r.u8()?)?)))
            .collect::<PyPrismResult<Vec<_>>>()?;
        terms.push((coefficient, factors));
    }
    r.finish()?;
    Ok(PauliObservable::from_terms(terms)?)
}
