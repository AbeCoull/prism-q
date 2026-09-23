//! Device calibration tables and their lowering to a [`NoiseModel`]: per-qubit
//! coherence times and readout rates, per-family gate durations and error
//! rates, a line-oriented text form, and illustrative presets in [`presets`].

use smallvec::smallvec;

use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, ReadoutError};

/// Coherence times and readout rates of one qubit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QubitCalibration {
    /// Relaxation time in seconds.
    pub t1: f64,
    /// Dephasing time in seconds, at most `2 * t1`.
    pub t2: f64,
    /// Probability a measured 0 reads out as 1.
    pub p01: f64,
    /// Probability a measured 1 reads out as 0.
    pub p10: f64,
}

/// Duration and error rate of one gate family or one qubit pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateCalibration {
    /// Gate duration in seconds, strictly positive.
    pub time: f64,
    /// Total depolarizing probability per gate, in `[0, 1]`.
    pub error: f64,
}

/// A device calibration table: one [`QubitCalibration`] per qubit, one
/// [`GateCalibration`] for the one-qubit and the two-qubit gate family, and
/// optional per-pair two-qubit entries that override the family on that pair.
///
/// Every constructor validates: `t2 <= 2 * t1`, probabilities in `[0, 1]`,
/// durations finite and positive, pairs distinct and inside the register. The
/// text form is read by [`DeviceCalibration::parse`]; the lowering to a circuit
/// is [`DeviceCalibration::to_noise_model`].
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, DeviceCalibration, Gate};
///
/// let text = "\
/// qubit 0 t1=120e-6 t2=80e-6 p01=0.02 p10=0.03
/// qubit 1 t1=95e-6 t2=110e-6 p01=0.01 p10=0.02
/// gate1q time=35e-9 error=3e-4
/// gate2q time=300e-9 error=8e-3
/// ";
/// let calibration = DeviceCalibration::parse(text)?;
///
/// let mut circuit = Circuit::new(2, 2);
/// circuit.add_gate(Gate::H, &[0]);
/// circuit.add_gate(Gate::Cx, &[0, 1]);
/// circuit.add_measure(0, 0);
/// circuit.add_measure(1, 1);
/// let noise = calibration.to_noise_model(&circuit)?;
/// assert_eq!(noise.after_gate[1].len(), 3);
/// # Ok::<(), prism_q::PrismError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCalibration {
    qubits: Vec<QubitCalibration>,
    gate1q: GateCalibration,
    gate2q: GateCalibration,
    pairs: Vec<(usize, usize, GateCalibration)>,
}

impl QubitCalibration {
    fn check(&self) -> std::result::Result<(), String> {
        check_time("t1", self.t1)?;
        check_time("t2", self.t2)?;
        if self.t2 > 2.0 * self.t1 {
            return Err(format!("t2 = {} exceeds twice t1 = {}", self.t2, self.t1));
        }
        check_probability("p01", self.p01)?;
        check_probability("p10", self.p10)
    }
}

impl GateCalibration {
    fn check(&self) -> std::result::Result<(), String> {
        check_time("time", self.time)?;
        check_probability("error", self.error)
    }
}

fn check_time(field: &str, value: f64) -> std::result::Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{field} = {value} must be finite and positive"));
    }
    Ok(())
}

fn check_probability(field: &str, value: f64) -> std::result::Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("{field} = {value} must be finite and in [0, 1]"));
    }
    Ok(())
}

fn invalid(message: String) -> PrismError {
    PrismError::InvalidParameter { message }
}

impl DeviceCalibration {
    /// Build a calibration for `qubits.len()` qubits with uniform gate families.
    ///
    /// # Errors
    ///
    /// Names the qubit and field of the first value outside its range.
    pub fn new(
        qubits: Vec<QubitCalibration>,
        gate1q: GateCalibration,
        gate2q: GateCalibration,
    ) -> Result<Self> {
        for (index, qubit) in qubits.iter().enumerate() {
            qubit
                .check()
                .map_err(|message| invalid(format!("qubit {index}: {message}")))?;
        }
        gate1q
            .check()
            .map_err(|message| invalid(format!("gate1q: {message}")))?;
        gate2q
            .check()
            .map_err(|message| invalid(format!("gate2q: {message}")))?;
        Ok(Self {
            qubits,
            gate1q,
            gate2q,
            pairs: Vec::new(),
        })
    }

    /// Give the unordered pair `(a, b)` its own two-qubit duration and error
    /// rate, replacing the family entry for gates on that pair in either order.
    ///
    /// # Errors
    ///
    /// Rejects a pair naming one qubit twice or a qubit outside the table, a
    /// pair already given, and a value outside its range.
    pub fn with_pair(mut self, a: usize, b: usize, gate: GateCalibration) -> Result<Self> {
        let (lo, hi) = (a.min(b), a.max(b));
        let prefix = format!("gate2q {a} {b}");
        if lo == hi {
            return Err(invalid(format!(
                "{prefix}: a pair needs two distinct qubits"
            )));
        }
        if hi >= self.qubits.len() {
            return Err(invalid(format!(
                "{prefix}: qubit {hi} is outside the {}-qubit table",
                self.qubits.len()
            )));
        }
        if self.pairs.iter().any(|&(x, y, _)| (x, y) == (lo, hi)) {
            return Err(invalid(format!("{prefix}: pair given twice")));
        }
        gate.check()
            .map_err(|message| invalid(format!("{prefix}: {message}")))?;
        self.pairs.push((lo, hi, gate));
        Ok(self)
    }

    /// Parse the line-oriented text form.
    ///
    /// One record per line, fields as `key=value` in any order, `#` to end of
    /// line is a comment, blank lines are skipped. Records:
    ///
    /// | Record | Fields | Notes |
    /// |---|---|---|
    /// | `qubit <n> ...` | `t1`, `t2` (seconds); `p01`, `p10` (default 0) | one per qubit, indices `0..N` in any order |
    /// | `gate1q ...` | `time` (seconds), `error` | exactly once |
    /// | `gate2q ...` | `time`, `error` | exactly once |
    /// | `gate2q <a> <b> ...` | `time`, `error` | optional, one per unordered pair |
    ///
    /// # Errors
    ///
    /// A [`PrismError::Parse`] naming the line and, where one is at fault, the
    /// field: a malformed token, an unknown or repeated key, a missing required
    /// field, a value outside its range, or a duplicate record. A qubit index
    /// left undefined below the highest one given is reported against the last
    /// line.
    pub fn parse(text: &str) -> Result<Self> {
        let mut qubits: Vec<Option<QubitCalibration>> = Vec::new();
        let mut gate1q = None;
        let mut gate2q = None;
        let mut pairs: Vec<(usize, usize, GateCalibration)> = Vec::new();
        let mut last_line = 0;

        for (offset, raw) in text.lines().enumerate() {
            let line = offset + 1;
            last_line = line;
            let body = raw.split('#').next().unwrap_or("").trim();
            if body.is_empty() {
                continue;
            }
            let fail = |message: String| PrismError::Parse { line, message };

            let mut tokens = body.split_whitespace();
            let kind = tokens.next().unwrap_or("");
            let mut indices: Vec<usize> = Vec::new();
            let mut fields: Vec<(&str, &str)> = Vec::new();
            for token in tokens {
                match token.split_once('=') {
                    Some((key, value)) => fields.push((key, value)),
                    None if fields.is_empty() => {
                        let index = token
                            .parse()
                            .map_err(|_| fail(format!("`{token}` is not a qubit index")))?;
                        indices.push(index);
                    }
                    None => {
                        return Err(fail(format!("expected key=value, got `{token}`")));
                    }
                }
            }

            match (kind, indices.as_slice()) {
                ("qubit", &[index]) => {
                    let get = FieldSet::new(fields, &["t1", "t2", "p01", "p10"], line)?;
                    let qubit = QubitCalibration {
                        t1: get.required("t1")?,
                        t2: get.required("t2")?,
                        p01: get.optional("p01")?.unwrap_or(0.0),
                        p10: get.optional("p10")?.unwrap_or(0.0),
                    };
                    qubit
                        .check()
                        .map_err(|message| fail(format!("qubit {index}: {message}")))?;
                    if qubits.len() <= index {
                        qubits.resize(index + 1, None);
                    }
                    if qubits[index].replace(qubit).is_some() {
                        return Err(fail(format!("qubit {index} given twice")));
                    }
                }
                ("gate1q", &[]) => {
                    let gate = gate_record(fields, "gate1q", line)?;
                    if gate1q.replace(gate).is_some() {
                        return Err(fail("gate1q given twice".into()));
                    }
                }
                ("gate2q", &[]) => {
                    let gate = gate_record(fields, "gate2q", line)?;
                    if gate2q.replace(gate).is_some() {
                        return Err(fail("gate2q given twice".into()));
                    }
                }
                ("gate2q", &[a, b]) => {
                    let prefix = format!("gate2q {a} {b}");
                    let gate = gate_record(fields, &prefix, line)?;
                    let (lo, hi) = (a.min(b), a.max(b));
                    if lo == hi {
                        return Err(fail(format!("{prefix}: a pair needs two distinct qubits")));
                    }
                    if pairs.iter().any(|&(x, y, _)| (x, y) == (lo, hi)) {
                        return Err(fail(format!("{prefix}: pair given twice")));
                    }
                    pairs.push((lo, hi, gate));
                }
                ("qubit", _) => {
                    return Err(fail(format!(
                        "`qubit` takes one index, got {}",
                        indices.len()
                    )));
                }
                ("gate1q" | "gate2q", _) => {
                    return Err(fail(format!(
                        "`{kind}` takes no index or a pair, got {}",
                        indices.len()
                    )));
                }
                _ => return Err(fail(format!("unknown record `{kind}`"))),
            }
        }

        let fail = |message: String| PrismError::Parse {
            line: last_line,
            message,
        };
        let qubits = qubits
            .iter()
            .enumerate()
            .map(|(index, qubit)| qubit.ok_or_else(|| fail(format!("qubit {index} is not given"))))
            .collect::<Result<Vec<_>>>()?;
        let gate1q = gate1q.ok_or_else(|| fail("gate1q is not given".into()))?;
        let gate2q = gate2q.ok_or_else(|| fail("gate2q is not given".into()))?;

        let mut calibration = Self::new(qubits, gate1q, gate2q)?;
        for (a, b, gate) in pairs {
            calibration = calibration
                .with_pair(a, b, gate)
                .map_err(|err| PrismError::Parse {
                    line: last_line,
                    message: err.to_string(),
                })?;
        }
        Ok(calibration)
    }

    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    pub fn qubits(&self) -> &[QubitCalibration] {
        &self.qubits
    }

    /// # Panics
    ///
    /// Panics if `qubit` is outside the table.
    pub fn qubit(&self, qubit: usize) -> &QubitCalibration {
        &self.qubits[qubit]
    }

    pub fn gate1q(&self) -> &GateCalibration {
        &self.gate1q
    }

    pub fn gate2q(&self) -> &GateCalibration {
        &self.gate2q
    }

    /// The entry a two-qubit gate on `a` and `b` uses, in either order: the
    /// pair's own if one was given, else the family's.
    pub fn gate2q_on(&self, a: usize, b: usize) -> &GateCalibration {
        let (lo, hi) = (a.min(b), a.max(b));
        self.pairs
            .iter()
            .find(|&&(x, y, _)| (x, y) == (lo, hi))
            .map_or(&self.gate2q, |(_, _, gate)| gate)
    }

    /// Lower the table onto `circuit`.
    ///
    /// After each gate, every target gets a
    /// [`NoiseChannel::ThermalRelaxation`] with its own `t1` and `t2` over the
    /// family's duration, relaxing to the ground state; then a nonzero family
    /// error adds one [`NoiseChannel::Depolarizing`] on a one-qubit target or one
    /// [`NoiseChannel::TwoQubitDepolarizing`] on the pair. Each measurement sets
    /// its classical bit's [`ReadoutError`] from the measured qubit, ideal when
    /// both rates are zero. Instructions other than gates and measurements carry
    /// no noise, as in the other [`NoiseModel`] constructors.
    ///
    /// # Errors
    ///
    /// Rejects a circuit wider than the table, a gate on more than two qubits,
    /// and everything [`NoiseModel::validate_for`] rejects.
    pub fn to_noise_model(&self, circuit: &Circuit) -> Result<NoiseModel> {
        if circuit.num_qubits > self.qubits.len() {
            return Err(invalid(format!(
                "circuit has {} qubits but the calibration covers {}",
                circuit.num_qubits,
                self.qubits.len()
            )));
        }
        let mut after_gate: Vec<Vec<NoiseEvent>> = vec![Vec::new(); circuit.instructions.len()];
        let mut readout = vec![None; circuit.num_classical_bits];

        for (idx, instr) in circuit.instructions.iter().enumerate() {
            match instr {
                Instruction::Gate { gate, targets } => {
                    let family = match targets.len() {
                        0 => continue,
                        1 => &self.gate1q,
                        2 => self.gate2q_on(targets[0], targets[1]),
                        arity => {
                            return Err(invalid(format!(
                                "gate `{}` at instruction {idx} acts on {arity} qubits and the \
                                 calibration has no family past two",
                                gate.name()
                            )));
                        }
                    };
                    let slot = &mut after_gate[idx];
                    for &qubit in targets.iter() {
                        let cal = &self.qubits[qubit];
                        slot.push(NoiseEvent {
                            channel: NoiseChannel::ThermalRelaxation {
                                t1: cal.t1,
                                t2: cal.t2,
                                gate_time: family.time,
                                excited_population: 0.0,
                            },
                            qubits: smallvec![qubit],
                        });
                    }
                    if family.error > 0.0 {
                        let (channel, qubits) = if targets.len() == 1 {
                            (
                                NoiseChannel::Depolarizing { p: family.error },
                                smallvec![targets[0]],
                            )
                        } else {
                            (
                                NoiseChannel::TwoQubitDepolarizing { p: family.error },
                                smallvec![targets[0], targets[1]],
                            )
                        };
                        slot.push(NoiseEvent { channel, qubits });
                    }
                }
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    let cal = &self.qubits[*qubit];
                    readout[*classical_bit] =
                        (cal.p01 > 0.0 || cal.p10 > 0.0).then_some(ReadoutError {
                            p01: cal.p01,
                            p10: cal.p10,
                        });
                }
                _ => {}
            }
        }

        let model = NoiseModel {
            after_gate,
            readout,
        };
        model.validate_for(circuit)?;
        Ok(model)
    }
}

/// The `key=value` fields of one record, read by name so a key outside
/// `allowed`, or given twice, is reported against its line.
struct FieldSet<'a> {
    fields: Vec<(&'a str, &'a str)>,
    line: usize,
}

impl<'a> FieldSet<'a> {
    fn new(fields: Vec<(&'a str, &'a str)>, allowed: &[&str], line: usize) -> Result<Self> {
        if let Some((bad, _)) = fields.iter().find(|(k, _)| !allowed.contains(k)) {
            return Err(PrismError::Parse {
                line,
                message: format!("unknown field `{bad}`"),
            });
        }
        Ok(Self { fields, line })
    }

    fn fail(&self, message: String) -> PrismError {
        PrismError::Parse {
            line: self.line,
            message,
        }
    }

    fn optional(&self, key: &str) -> Result<Option<f64>> {
        let mut found = None;
        for (k, value) in &self.fields {
            if *k != key {
                continue;
            }
            if found.is_some() {
                return Err(self.fail(format!("field `{key}` given twice")));
            }
            let parsed: f64 = value
                .parse()
                .map_err(|_| self.fail(format!("field `{key}`: `{value}` is not a number")))?;
            found = Some(parsed);
        }
        Ok(found)
    }

    fn required(&self, key: &str) -> Result<f64> {
        self.optional(key)?
            .ok_or_else(|| self.fail(format!("field `{key}` is missing")))
    }
}

fn gate_record(fields: Vec<(&str, &str)>, prefix: &str, line: usize) -> Result<GateCalibration> {
    let get = FieldSet::new(fields, &["time", "error"], line)?;
    let gate = GateCalibration {
        time: get.required("time")?,
        error: get.required("error")?,
    };
    gate.check().map_err(|message| PrismError::Parse {
        line,
        message: format!("{prefix}: {message}"),
    })?;
    Ok(gate)
}

/// Calibrations with typical magnitudes for a technology class.
///
/// The values are illustrative orders of magnitude for exploring how a noise
/// model of that class behaves, not a measured device: every qubit is given the
/// same numbers and no pair entry is set. Edit the text form or build a
/// [`DeviceCalibration`] directly when a specific device is the target.
pub mod presets {
    use super::{DeviceCalibration, GateCalibration, QubitCalibration};

    fn uniform(
        num_qubits: usize,
        qubit: QubitCalibration,
        gate1q: GateCalibration,
        gate2q: GateCalibration,
    ) -> DeviceCalibration {
        DeviceCalibration::new(vec![qubit; num_qubits], gate1q, gate2q)
            .expect("preset values are inside every range")
    }

    /// Fixed-frequency transmon magnitudes: `t1` 100 us, `t2` 80 us, readout
    /// 1% and 3%, 30 ns one-qubit gates at 3e-4, 300 ns two-qubit gates at
    /// 8e-3.
    pub fn superconducting_transmon(num_qubits: usize) -> DeviceCalibration {
        uniform(
            num_qubits,
            QubitCalibration {
                t1: 100e-6,
                t2: 80e-6,
                p01: 0.01,
                p10: 0.03,
            },
            GateCalibration {
                time: 30e-9,
                error: 3e-4,
            },
            GateCalibration {
                time: 300e-9,
                error: 8e-3,
            },
        )
    }

    /// Trapped-ion magnitudes: `t1` 10 s, `t2` 1 s, readout 0.3% each way,
    /// 10 us one-qubit gates at 1e-4, 200 us two-qubit gates at 5e-3.
    pub fn trapped_ion(num_qubits: usize) -> DeviceCalibration {
        uniform(
            num_qubits,
            QubitCalibration {
                t1: 10.0,
                t2: 1.0,
                p01: 0.003,
                p10: 0.003,
            },
            GateCalibration {
                time: 10e-6,
                error: 1e-4,
            },
            GateCalibration {
                time: 200e-6,
                error: 5e-3,
            },
        )
    }

    /// Neutral-atom magnitudes: `t1` 4 s, `t2` 1 ms, readout 1% and 2%,
    /// 1 us one-qubit gates at 5e-4, 500 ns two-qubit gates at 1e-2.
    pub fn neutral_atom(num_qubits: usize) -> DeviceCalibration {
        uniform(
            num_qubits,
            QubitCalibration {
                t1: 4.0,
                t2: 1e-3,
                p01: 0.01,
                p10: 0.02,
            },
            GateCalibration {
                time: 1e-6,
                error: 5e-4,
            },
            GateCalibration {
                time: 500e-9,
                error: 1e-2,
            },
        )
    }
}
