//! Rewrite of a dynamic circuit's resets, reused measurements and classically guarded
//! Paulis into an equivalent circuit with terminal measurements, for compiled sampling.

use crate::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::noise::{NoiseEvent, NoiseModel};

use super::incompatible;

/// Convert reset reuse into fresh qubit aliases, copy the record of a measured qubit that
/// is used again onto a fresh alias, and move measurements to the end. A guarded Pauli
/// becomes that Pauli controlled by the alias of each record its guard reads, the
/// deferred-measurement form of classical feed-forward.
pub(crate) fn defer_measure_reset_circuit(circuit: &Circuit) -> Result<Circuit> {
    unguarded(defer_guarded_circuit(circuit)?)
}

/// [`defer_measure_reset_circuit`] on a Clifford+T circuit. A guard still applies only a
/// Pauli, so the rewrite keeps the T count and adds only Clifford gates.
pub(crate) fn defer_clifford_t_circuit(circuit: &Circuit) -> Result<Circuit> {
    let mut walk = DeferralWalk::new(circuit, None, true)?;
    walk.run(&circuit.instructions)?;
    unguarded(walk.finish())
}

fn unguarded(deferred: DeferredCircuit) -> Result<Circuit> {
    if !deferred.guarded.is_empty() {
        return Err(incompatible(
            "deferred measurement sampling applies a guard as a parity of measured \
             bits, and a register comparison that reads two measured bits is not one",
        ));
    }
    Ok(deferred.circuit)
}

/// [`defer_measure_reset_circuit`] keeping the guarded Paulis no controlled Pauli applies,
/// those under a comparison of several measured bits, as [`DeferredCircuit::guarded`]
/// entries over a model with no events.
pub(crate) fn defer_guarded_circuit(circuit: &Circuit) -> Result<DeferredCircuit> {
    let mut walk = DeferralWalk::new(circuit, None, false)?;
    walk.run(&circuit.instructions)?;
    Ok(walk.finish())
}

/// A dynamic circuit rewritten by [`defer_noisy_circuit`].
pub(crate) struct DeferredCircuit {
    /// Gates on the aliases, then one terminal measurement per record in record order.
    pub(crate) circuit: Circuit,
    /// The events that fire on every shot, one slot per instruction of `circuit`, and the
    /// source model's readout on the bits no guard reads. A bit a guard reads takes its
    /// readout error as an `X` on each alias measured into it, where the measurement
    /// writes it, so the guard and the record both see the flip: an every-shot event when
    /// `p01 == p10`, else a [`GuardedKind::Readout`] entry.
    pub(crate) noise: NoiseModel,
    /// Guarded Paulis and events of guarded bodies in execution order, each landing only
    /// on the shots whose guard holds.
    pub(crate) guarded: Vec<GuardedEvent>,
}

/// A one-qubit Pauli that lands after the compiled sample, on `qubit` after instruction
/// `anchor` of the deferred circuit.
pub(crate) struct GuardedEvent {
    pub(crate) anchor: usize,
    pub(crate) guard: Guard,
    pub(crate) qubit: usize,
    pub(crate) kind: GuardedKind,
}

/// The records a guarded entry reads, each one a measurement of the deferred circuit.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Guard {
    /// Holds where the parity of the records, XOR `invert`, is one.
    Parity {
        records: SmallVec<[usize; 4]>,
        invert: bool,
    },
    /// Holds where every record equals its wanted bit, XOR `negate`.
    All {
        terms: SmallVec<[(usize, bool); 4]>,
        negate: bool,
    },
}

impl Guard {
    /// Whether the guard holds on a shot whose record `r` reads `bit(r)`.
    #[inline(always)]
    pub(crate) fn holds(&self, bit: impl Fn(usize) -> bool) -> bool {
        match self {
            Guard::Parity { records, invert } => records
                .iter()
                .fold(*invert, |acc, &record| acc ^ bit(record)),
            Guard::All { terms, negate } => {
                terms.iter().all(|&(record, want)| bit(record) == want) ^ negate
            }
        }
    }

    fn is_constant(&self, value: bool) -> bool {
        matches!(self, Guard::Parity { records, invert } if records.is_empty() && *invert == value)
    }
}

/// What a [`GuardedEvent`] lands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GuardedKind {
    /// A guarded Pauli gate, letter 1 `X`, 2 `Y`, 3 `Z`, on every shot the guard holds.
    Gate(u8),
    /// A guarded body slot's Pauli channel `(px, py, pz)`, drawn per shot.
    Noise((f64, f64, f64)),
    /// An `X` on the alias measured into `record`, drawn per shot at `p01` where the record
    /// reads 0 and `p10` where it reads 1.
    Readout { record: usize, p01: f64, p10: f64 },
}

/// [`defer_measure_reset_circuit`] carrying `noise` onto the rewrite. An event on a
/// measured alias drops, since its record already holds the value the shot keeps. A
/// measurement whose qubit is used again before a reset copies its record onto a fresh
/// alias instead, so the qubit stays live and noise after the copy reaches it alone.
pub(crate) fn defer_noisy_circuit(
    circuit: &Circuit,
    noise: &NoiseModel,
) -> Result<DeferredCircuit> {
    let mut walk = DeferralWalk::new(circuit, Some(noise), false)?;
    noise.ensure_only_pauli_channels()?;
    walk.run(&circuit.instructions)?;
    Ok(walk.finish())
}

struct DeferralWalk<'a> {
    num_classical_bits: usize,
    aliases: Vec<usize>,
    measured_aliases: Vec<bool>,
    next_qubit: usize,
    deferred_measurements: Vec<(usize, usize)>,
    /// Record each classical bit last took, `None` until a measurement writes it.
    bit_records: Vec<Option<usize>>,
    /// Readout flip rates `(p01, p10)` of each bit a guard reads, applied where a
    /// measurement writes it.
    flips_at_write: Vec<Option<(f64, f64)>>,
    transformed: Circuit,
    noise: Option<&'a NoiseModel>,
    after_gate: Vec<Vec<NoiseEvent>>,
    guarded: Vec<GuardedEvent>,
    /// Top-level measurements that copy their record onto a fresh alias, leaving the
    /// measured alias live for the instructions that touch it later.
    copied_measurements: Vec<bool>,
}

impl<'a> DeferralWalk<'a> {
    /// `with_t` admits Clifford+T gates outside guards in place of Clifford gates alone.
    fn new(circuit: &Circuit, noise: Option<&'a NoiseModel>, with_t: bool) -> Result<Self> {
        if with_t && !circuit.is_clifford_plus_t() {
            return Err(incompatible(
                "deferred measurement sampling requires Clifford+T gates",
            ));
        }
        if !with_t && !circuit.is_clifford_only() {
            return Err(incompatible(
                "deferred measurement sampling requires Clifford gates",
            ));
        }

        let has_measurements = circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Measure { .. }));
        if !has_measurements {
            return Err(incompatible(
                "deferred measurement sampling requires measurements",
            ));
        }

        let flips_at_write = match noise {
            Some(noise) => readout_flips_at_write(circuit, noise)?,
            None => Vec::new(),
        };
        let copied_measurements = reused_measurements(circuit, noise.is_some());
        Ok(Self {
            num_classical_bits: circuit.num_classical_bits,
            aliases: (0..circuit.num_qubits).collect(),
            measured_aliases: vec![false; circuit.num_qubits],
            next_qubit: circuit.num_qubits,
            deferred_measurements: Vec::new(),
            bit_records: vec![None; circuit.num_classical_bits],
            flips_at_write,
            transformed: Circuit::new(circuit.num_qubits, circuit.num_classical_bits),
            noise,
            after_gate: Vec::new(),
            guarded: Vec::new(),
            copied_measurements,
        })
    }

    /// Rewrite the top-level `instructions`, whose guarded noise slots follow their own.
    fn run(&mut self, instructions: &[Instruction]) -> Result<()> {
        let mut nested = instructions.len();
        for (slot, inst) in instructions.iter().enumerate() {
            match inst {
                Instruction::Gate { gate, targets } => {
                    let mapped =
                        map_deferred_targets(targets, &self.aliases, &self.measured_aliases)?;
                    self.transformed.instructions.push(Instruction::Gate {
                        gate: gate.clone(),
                        targets: mapped,
                    });
                }
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    if *qubit >= self.aliases.len() {
                        return Err(PrismError::InvalidQubit {
                            index: *qubit,
                            register_size: self.aliases.len(),
                        });
                    }
                    if *classical_bit >= self.num_classical_bits {
                        return Err(PrismError::InvalidClassicalBit {
                            index: *classical_bit,
                            register_size: self.num_classical_bits,
                        });
                    }
                    let alias = self.aliases[*qubit];
                    let record_alias = if self.copied_measurements.get(slot) == Some(&true) {
                        let copy = self.next_qubit;
                        self.next_qubit += 1;
                        self.measured_aliases.push(true);
                        self.transformed.num_qubits = self.next_qubit;
                        self.transformed.add_gate(Gate::Cx, &[alias, copy]);
                        copy
                    } else {
                        if self.noise.is_some() && self.measured_aliases[alias] {
                            return Err(incompatible(
                                "deferred noisy sampling requires reset before \
                                 measuring a measured qubit again, since noise between \
                                 the two measurements reaches the second alone",
                            ));
                        }
                        self.measured_aliases[alias] = true;
                        alias
                    };
                    let record = self.deferred_measurements.len();
                    self.bit_records[*classical_bit] = Some(record);
                    self.deferred_measurements
                        .push((record_alias, *classical_bit));
                    if let Some((p01, p10)) =
                        self.flips_at_write.get(*classical_bit).copied().flatten()
                    {
                        let anchor = self.anchor();
                        if p01 == p10 {
                            self.after_gate[anchor].push(NoiseEvent::pauli(
                                record_alias,
                                p01,
                                0.0,
                                0.0,
                            ));
                        } else {
                            self.guarded.push(GuardedEvent {
                                anchor,
                                guard: Guard::Parity {
                                    records: SmallVec::new(),
                                    invert: true,
                                },
                                qubit: record_alias,
                                kind: GuardedKind::Readout { record, p01, p10 },
                            });
                        }
                    }
                }
                Instruction::Reset { qubit } => {
                    if *qubit >= self.aliases.len() {
                        return Err(PrismError::InvalidQubit {
                            index: *qubit,
                            register_size: self.aliases.len(),
                        });
                    }
                    self.aliases[*qubit] = self.next_qubit;
                    self.next_qubit += 1;
                    self.measured_aliases.push(false);
                    self.transformed.num_qubits = self.next_qubit;
                }
                Instruction::Barrier { qubits } => {
                    let mut mapped = SmallVec::<[usize; 4]>::with_capacity(qubits.len());
                    for &q in qubits {
                        if q >= self.aliases.len() {
                            return Err(PrismError::InvalidQubit {
                                index: q,
                                register_size: self.aliases.len(),
                            });
                        }
                        mapped.push(self.aliases[q]);
                    }
                    self.transformed
                        .instructions
                        .push(Instruction::Barrier { qubits: mapped });
                }
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } => {
                    require_guarded_pauli(gate)?;
                    let guard = self.guard(condition)?;
                    self.push_guarded_gate(gate, targets, &guard)?;
                    self.carry_guarded(nested, &guard)?;
                    nested += 1;
                }
                Instruction::Region(region) => {
                    for body_inst in region.body() {
                        match body_inst {
                            Instruction::Gate { gate, .. } => require_guarded_pauli(gate)?,
                            Instruction::Barrier { .. } => {}
                            _ => {
                                return Err(incompatible(
                                    "deferred measurement sampling does not support a \
                                     guarded body holding a measurement, reset, save \
                                     point or nested guard",
                                ));
                            }
                        }
                    }
                    let guard = self.guard(region.condition())?;
                    for (offset, body_inst) in region.body().iter().enumerate() {
                        if let Instruction::Gate { gate, targets } = body_inst {
                            self.push_guarded_gate(gate, targets, &guard)?;
                        }
                        self.carry_guarded(nested + offset, &guard)?;
                    }
                    nested += region.noise_slots();
                }
                Instruction::Save { label, .. } => {
                    return Err(incompatible(format!(
                        "save point `{label}` needs a state to read, and shot sampling \
                         reorders measurements rather than holding one"
                    )));
                }
            }
            self.carry(slot);
        }
        Ok(())
    }

    /// The guard deciding `condition` against the bits the measurements so far wrote. A
    /// bit no measurement wrote reads 0 and drops out, so a register comparison is a parity
    /// while it reads at most one written bit, and a conjunction of records past that.
    fn guard(&self, condition: &ClassicalCondition) -> Result<Guard> {
        let record = |bit: usize| {
            self.bit_records
                .get(bit)
                .copied()
                .ok_or(PrismError::InvalidClassicalBit {
                    index: bit,
                    register_size: self.num_classical_bits,
                })
        };
        let parity =
            |records: SmallVec<[usize; 4]>, invert: bool| Guard::Parity { records, invert };
        match condition {
            ClassicalCondition::BitIsOne(bit) => {
                Ok(parity(record(*bit)?.into_iter().collect(), false))
            }
            ClassicalCondition::BitIsZero(bit) => {
                Ok(parity(record(*bit)?.into_iter().collect(), true))
            }
            ClassicalCondition::Parity { bits, expected } => {
                let mut records = SmallVec::new();
                for &bit in bits.iter() {
                    records.extend(record(bit)?);
                }
                Ok(parity(records, !expected))
            }
            ClassicalCondition::RegisterEquals {
                offset,
                size,
                value,
            }
            | ClassicalCondition::RegisterNotEquals {
                offset,
                size,
                value,
            } => {
                let not_equals = matches!(condition, ClassicalCondition::RegisterNotEquals { .. });
                let mut feasible = *size >= 64 || value >> size == 0;
                let mut written = SmallVec::<[(usize, bool); 4]>::new();
                for i in 0..*size {
                    let want = i < 64 && (value >> i) & 1 == 1;
                    match record(offset.saturating_add(i))? {
                        None => feasible &= !want,
                        Some(r) => written.push((r, want)),
                    }
                }
                Ok(match (feasible, written.as_slice()) {
                    (false, _) => parity(SmallVec::new(), not_equals),
                    (true, []) => parity(SmallVec::new(), !not_equals),
                    (true, &[(r, want)]) => parity(SmallVec::from_slice(&[r]), want == not_equals),
                    (true, _) => Guard::All {
                        terms: written,
                        negate: not_equals,
                    },
                })
            }
        }
    }

    /// Apply Pauli `gate` on `targets[0]` under `guard`: a parity as controlled Paulis in the
    /// deferred circuit, a conjunction as a [`GuardedKind::Gate`] entry after the last
    /// instruction emitted.
    fn push_guarded_gate(
        &mut self,
        gate: &Gate,
        targets: &SmallVec<[usize; 4]>,
        guard: &Guard,
    ) -> Result<()> {
        let letter = match (guard, gate) {
            (Guard::Parity { records, invert }, _) => {
                return self.push_guarded_pauli(gate, targets, records, *invert);
            }
            (Guard::All { .. }, Gate::Id) => return Ok(()),
            (Guard::All { .. }, Gate::X) => 1,
            (Guard::All { .. }, Gate::Y) => 2,
            (Guard::All { .. }, _) => 3,
        };
        let qubit = map_deferred_targets(targets, &self.aliases, &self.measured_aliases)?[0];
        let anchor = self.anchor();
        self.guarded.push(GuardedEvent {
            anchor,
            guard: guard.clone(),
            qubit,
            kind: GuardedKind::Gate(letter),
        });
        Ok(())
    }

    /// Emit Pauli `gate` on `targets[0]` under the guard `(records, invert)`: once
    /// unconditionally when `invert` holds, then once controlled by each record's alias.
    fn push_guarded_pauli(
        &mut self,
        gate: &Gate,
        targets: &SmallVec<[usize; 4]>,
        records: &[usize],
        invert: bool,
    ) -> Result<()> {
        if matches!(gate, Gate::Id) {
            return Ok(());
        }
        let target = map_deferred_targets(targets, &self.aliases, &self.measured_aliases)?[0];
        if invert {
            self.transformed.add_gate(gate.clone(), &[target]);
        }
        if records.is_empty() {
            return Ok(());
        }
        let (controlled, conjugate) = match gate {
            Gate::X => (Gate::Cx, false),
            Gate::Z => (Gate::Cz, false),
            _ => (Gate::Cx, true),
        };
        if conjugate {
            self.transformed.add_gate(Gate::Sdg, &[target]);
        }
        for &record in records {
            let control = self.deferred_measurements[record].0;
            self.transformed
                .add_gate(controlled.clone(), &[control, target]);
        }
        if conjugate {
            self.transformed.add_gate(Gate::S, &[target]);
        }
        Ok(())
    }

    /// Index of the last instruction emitted, which an event carried now follows. An
    /// empty barrier stands in while nothing is emitted.
    fn anchor(&mut self) -> usize {
        if self.transformed.instructions.is_empty() {
            self.transformed.instructions.push(Instruction::Barrier {
                qubits: SmallVec::new(),
            });
        }
        let len = self.transformed.instructions.len();
        self.after_gate.resize_with(len, Vec::new);
        len - 1
    }

    /// `event` on the aliases its qubits hold now, `None` once every target is measured.
    /// A pair losing one target keeps the marginal on the other, 4p/15 per letter.
    fn live_event(&self, event: &NoiseEvent) -> Option<NoiseEvent> {
        let live = |q: usize| {
            let alias = self.aliases[q];
            (!self.measured_aliases[alias]).then_some(alias)
        };
        if let Some((q0, q1, p)) = event.pauli_pair() {
            return match (live(q0), live(q1)) {
                (Some(a0), Some(a1)) => Some(NoiseEvent {
                    channel: event.channel.clone(),
                    qubits: SmallVec::from_slice(&[a0, a1]),
                }),
                (Some(alias), None) | (None, Some(alias)) => {
                    let letter = p * 4.0 / 15.0;
                    Some(NoiseEvent::pauli(alias, letter, letter, letter))
                }
                (None, None) => None,
            };
        }
        live(event.qubit()).map(|alias| NoiseEvent {
            channel: event.channel.clone(),
            qubits: SmallVec::from_slice(&[alias]),
        })
    }

    /// Carry the events of top-level slot `slot` onto the last instruction emitted.
    fn carry(&mut self, slot: usize) {
        let Some(noise) = self.noise else {
            return;
        };
        for event in &noise.after_gate[slot] {
            if let Some(live) = self.live_event(event) {
                let anchor = self.anchor();
                self.after_gate[anchor].push(live);
            }
        }
    }

    /// Carry the events of guarded slot `slot`, which fire on the shots `guard` holds:
    /// into [`DeferredCircuit::guarded`], or among the every-shot events when the guard
    /// reads no record and so always holds.
    fn carry_guarded(&mut self, slot: usize, guard: &Guard) -> Result<()> {
        let Some(events) = self.noise.and_then(|noise| noise.after_gate.get(slot)) else {
            return Ok(());
        };
        if guard.is_constant(false) {
            return Ok(());
        }
        for event in events {
            let Some(probs) = event.channel.as_pauli() else {
                return Err(incompatible(
                    "deferred noisy sampling draws a guarded body's events one qubit \
                     at a time, and a two-qubit channel in a body slot needs a \
                     per-shot route",
                ));
            };
            let alias = self.aliases[event.qubit()];
            if self.measured_aliases[alias] || event.channel.is_inert() {
                continue;
            }
            let anchor = self.anchor();
            if guard.is_constant(true) {
                self.after_gate[anchor].push(NoiseEvent {
                    channel: event.channel.clone(),
                    qubits: SmallVec::from_slice(&[alias]),
                });
            } else {
                self.guarded.push(GuardedEvent {
                    anchor,
                    guard: guard.clone(),
                    qubit: alias,
                    kind: GuardedKind::Noise(probs),
                });
            }
        }
        Ok(())
    }

    fn finish(mut self) -> DeferredCircuit {
        self.transformed.num_qubits = self.next_qubit;
        self.transformed
            .instructions
            .reserve(self.deferred_measurements.len());
        for (qubit, classical_bit) in self.deferred_measurements {
            self.transformed.instructions.push(Instruction::Measure {
                qubit,
                classical_bit,
            });
        }
        self.after_gate
            .resize_with(self.transformed.instructions.len(), Vec::new);
        let readout = match self.noise {
            Some(noise) => {
                let mut readout = noise.readout.clone();
                for (entry, flip) in readout.iter_mut().zip(&self.flips_at_write) {
                    if flip.is_some() {
                        *entry = None;
                    }
                }
                readout
            }
            None => Vec::new(),
        };
        DeferredCircuit {
            circuit: self.transformed,
            noise: NoiseModel {
                after_gate: self.after_gate,
                readout,
            },
            guarded: self.guarded,
        }
    }
}

/// Mark each top-level measurement whose qubit a later gate, guarded gate or region touches
/// before the qubit's next reset; on the noisy walk a later measurement counts too. A marked
/// measurement becomes `CX` onto a fresh alias measured at the end, the deferred-measurement
/// form, which stays exact for any later use of the qubit because the copy only controls.
fn reused_measurements(circuit: &Circuit, noisy: bool) -> Vec<bool> {
    let mut touched = vec![false; circuit.num_qubits];
    let mut marked = vec![false; circuit.instructions.len()];
    let touch = |touched: &mut [bool], qubits: &[usize]| {
        for &q in qubits {
            if let Some(t) = touched.get_mut(q) {
                *t = true;
            }
        }
    };
    for (slot, inst) in circuit.instructions.iter().enumerate().rev() {
        match inst {
            Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => {
                touch(&mut touched, targets)
            }
            Instruction::Region(region) => touch(&mut touched, region.qubits()),
            Instruction::Measure { qubit, .. } => {
                if let Some(t) = touched.get_mut(*qubit) {
                    marked[slot] = *t;
                    *t |= noisy;
                }
            }
            Instruction::Reset { qubit } => {
                if let Some(t) = touched.get_mut(*qubit) {
                    *t = false;
                }
            }
            Instruction::Barrier { .. } | Instruction::Save { .. } => {}
        }
    }
    marked
}

/// The readout flip rates `(p01, p10)` of each classical bit a guard reads, for the bits
/// whose readout error can flip them. A measured alias only ever controls later gates, so
/// an `X` on it flips the record and every guard reading it and leaves the rest of the state
/// alone. Equal rates fire that `X` whatever the outcome; unequal ones pick the rate by the
/// record's value after the compiled sample.
fn readout_flips_at_write(
    circuit: &Circuit,
    noise: &NoiseModel,
) -> Result<Vec<Option<(f64, f64)>>> {
    let read = circuit.condition_bits();
    let mut flips = vec![None; read.len()];
    for (bit, err) in noise.readout.iter().enumerate() {
        let Some(err) = err.as_ref().filter(|err| !err.is_inert()) else {
            continue;
        };
        if !read.get(bit).copied().unwrap_or(false) {
            continue;
        }
        flips[bit] = Some((err.p01, err.p10));
    }
    Ok(flips)
}

/// Reject a guarded gate other than a Pauli, which no controlled Clifford applies.
fn require_guarded_pauli(gate: &Gate) -> Result<()> {
    if matches!(gate, Gate::Id | Gate::X | Gate::Y | Gate::Z) {
        return Ok(());
    }
    Err(incompatible(format!(
        "deferred measurement sampling does not support classical conditionals on `{}`: \
         it applies a guarded Pauli (`X`, `Y`, `Z`, `I`) as Paulis controlled by the \
         measured qubits",
        gate.name()
    )))
}

fn map_deferred_targets(
    targets: &SmallVec<[usize; 4]>,
    aliases: &[usize],
    measured_aliases: &[bool],
) -> Result<SmallVec<[usize; 4]>> {
    let mut mapped = SmallVec::<[usize; 4]>::with_capacity(targets.len());
    for &target in targets {
        if target >= aliases.len() {
            return Err(PrismError::InvalidQubit {
                index: target,
                register_size: aliases.len(),
            });
        }
        let alias = aliases[target];
        if measured_aliases[alias] {
            return Err(incompatible(
                "deferred measurement sampling requires reset before reusing a measured qubit",
            ));
        }
        mapped.push(alias);
    }
    Ok(mapped)
}

#[cfg(test)]
#[path = "defer_tests.rs"]
mod tests;
