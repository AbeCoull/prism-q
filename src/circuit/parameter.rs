//! Parameter slots over a circuit's rotation angles, shared by gradient and
//! binding consumers.

use std::mem::Discriminant;

use super::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

/// Binds one rotation gate instruction to a slot in the parameter vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamLink {
    /// Index into [`Circuit::instructions`].
    pub instruction: usize,
    /// Slot in the parameter vector this gate reads.
    pub slot: usize,
}

/// Parameter slots over the rotation angles of a circuit.
///
/// A slot may drive several instructions (weight sharing): [`bind`](Self::bind)
/// writes one angle to each, and the adjoint gradient accumulates their
/// contributions into one entry. Arity is declared at construction rather than
/// inferred from the links, so a value vector of the wrong length is rejected
/// even when the trailing slots carry no link.
///
/// Bindable gates are `Rx`, `Ry`, `Rz`, `Rzz`, `P`, and `PauliRot`, the `Gate`
/// variants carrying a rotation angle.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Parameters {
    links: Vec<ParamLink>,
    num_slots: usize,
    /// Slot names, empty when the set is positional only. OpenQASM `input`
    /// declarations are named, so export and the parser share these.
    names: Vec<String>,
    /// Gate kind and targets at each linked index when the set was built.
    /// Links are instruction indices, so an edited circuit would otherwise
    /// rebind silently; [`validate`](Parameters::validate) compares this.
    shape: Vec<(Discriminant<Gate>, SmallVec<[usize; 4]>)>,
}

impl Parameters {
    /// Declare `num_slots` slots with no links yet.
    pub fn new(num_slots: usize) -> Self {
        Self {
            num_slots,
            ..Default::default()
        }
    }

    /// Build from explicit links over a vector of `num_slots` entries.
    pub fn from_links(links: Vec<ParamLink>, num_slots: usize) -> Self {
        Self {
            links,
            num_slots,
            ..Default::default()
        }
    }

    /// Name the slots, in slot order. Names are what OpenQASM `input`
    /// declarations carry and what export emits.
    ///
    /// # Panics
    /// Panics if `names` is not exactly [`num_slots`](Self::num_slots) long.
    pub fn with_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        assert_eq!(
            names.len(),
            self.num_slots,
            "expected {} slot names, got {}",
            self.num_slots,
            names.len()
        );
        self.names = names;
        self
    }

    /// Name of `slot`, or `None` when the set is positional only.
    pub fn name_of(&self, slot: usize) -> Option<&str> {
        self.names.get(slot).map(String::as_str)
    }

    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Record the gate kind and targets each link points at, so a later
    /// [`validate`](Self::validate) against an edited circuit fails loudly
    /// rather than binding the wrong gates.
    pub fn pinned_to(mut self, circuit: &Circuit) -> Self {
        self.shape = self
            .links
            .iter()
            .filter_map(|link| match circuit.instructions.get(link.instruction) {
                Some(Instruction::Gate { gate, targets }) => {
                    Some((std::mem::discriminant(gate), targets.clone()))
                }
                _ => None,
            })
            .collect();
        if self.shape.len() != self.links.len() {
            self.shape.clear();
        }
        self
    }

    /// Give every bindable gate its own slot, in circuit order.
    pub fn all_rotations(circuit: &Circuit) -> Self {
        let mut links = Vec::new();
        for (i, inst) in circuit.instructions.iter().enumerate() {
            if let Instruction::Gate { gate, .. } = inst
                && gate.pauli_generator().is_some()
            {
                links.push(ParamLink {
                    instruction: i,
                    slot: links.len(),
                });
            }
        }
        let num_slots = links.len();
        Self {
            links,
            num_slots,
            ..Default::default()
        }
        .pinned_to(circuit)
    }

    /// Record that `instruction` reads `slot`, widening the declared slot count
    /// to cover it. For builders accumulating links before the count is known.
    pub(super) fn link_growing(&mut self, instruction: usize, slot: usize) {
        self.num_slots = self.num_slots.max(slot + 1);
        self.links.push(ParamLink { instruction, slot });
    }

    /// Record that `instruction` reads `slot`.
    ///
    /// # Panics
    /// Panics if `slot` is not below the declared slot count.
    pub fn link(&mut self, instruction: usize, slot: usize) {
        assert!(
            slot < self.num_slots,
            "slot {} out of bounds (parameter set declares {} slots)",
            slot,
            self.num_slots
        );
        self.links.push(ParamLink { instruction, slot });
    }

    pub fn links(&self) -> &[ParamLink] {
        &self.links
    }

    /// Length of the value vector [`bind`](Self::bind) expects.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// True when no instruction is linked. A set may still declare slots.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Check every link against `circuit` without binding.
    ///
    /// A declared slot that no instruction reads is accepted: OpenQASM allows an
    /// `input` the body never uses, and a sweep may range over a superset of the
    /// circuit's parameters. Use [`unread_slots`](Self::unread_slots) to report
    /// them.
    ///
    /// # Errors
    /// Returns [`PrismError::InvalidParameter`] when a link points past the end
    /// of the instruction stream, at a non-gate instruction, or at a gate
    /// carrying no angle, or references a slot outside the declared value vector.
    pub fn validate(&self, circuit: &Circuit) -> Result<()> {
        let n = circuit.instructions.len();
        for link in &self.links {
            if link.slot >= self.num_slots {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "parameter link references slot {} but the parameter set declares {} slots",
                        link.slot, self.num_slots
                    ),
                });
            }
            if link.instruction >= n {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "parameter link references instruction {} but the circuit has {n} instructions",
                        link.instruction
                    ),
                });
            }
            match &circuit.instructions[link.instruction] {
                Instruction::Gate { gate, .. } if gate.pauli_generator().is_some() => {}
                Instruction::Gate { gate, .. } => {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "instruction {} (`{}`) carries no bindable angle; bindable gates are rx, ry, rz, rzz, p, pauli_rot",
                            link.instruction,
                            gate.name()
                        ),
                    });
                }
                _ => {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "parameter link references instruction {} which is not a gate",
                            link.instruction
                        ),
                    });
                }
            }
        }

        if !self.shape.is_empty() {
            for (link, (kind, targets)) in self.links.iter().zip(&self.shape) {
                let Instruction::Gate { gate, targets: at } =
                    &circuit.instructions[link.instruction]
                else {
                    unreachable!("link validated as a gate above")
                };
                if std::mem::discriminant(gate) != *kind || at.as_slice() != targets.as_slice() {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "instruction {} no longer holds the gate this parameter set was built against; the circuit was edited after the links were recorded",
                            link.instruction
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Declared slots that no instruction reads, whose bound values are
    /// discarded. Legal, but usually a mistake worth surfacing.
    pub fn unread_slots(&self) -> Vec<usize> {
        let mut used = vec![false; self.num_slots];
        for link in &self.links {
            used[link.slot] = true;
        }
        (0..self.num_slots).filter(|s| !used[*s]).collect()
    }

    /// Write `values` into a copy of `template`.
    ///
    /// # Errors
    /// Returns [`PrismError::InvalidParameter`] when `values.len()` differs from
    /// [`num_slots`](Self::num_slots), when any value is not finite, or when
    /// [`validate`](Self::validate) rejects the links.
    pub fn bind(&self, template: &Circuit, values: &[f64]) -> Result<Circuit> {
        let mut out = template.clone();
        self.bind_into(template, values, &mut out)?;
        Ok(out)
    }

    /// Bind into an existing circuit, reusing its allocations.
    ///
    /// `out` is overwritten with `template` and then patched, so a sweep can
    /// hold one buffer across every point instead of allocating per binding.
    ///
    /// # Errors
    /// Same conditions as [`bind`](Self::bind).
    pub fn bind_into(&self, template: &Circuit, values: &[f64], out: &mut Circuit) -> Result<()> {
        self.check_values(values)?;
        self.validate(template)?;

        out.num_qubits = template.num_qubits;
        out.num_classical_bits = template.num_classical_bits;
        out.instructions.clone_from(&template.instructions);
        self.write_angles(out, values);
        Ok(())
    }

    /// Check `values` against the declared arity without touching a circuit.
    ///
    /// # Errors
    /// Same arity and finiteness conditions as [`bind`](Self::bind); the link
    /// checks belong to [`validate`](Self::validate) and are not repeated here.
    pub(crate) fn check_values(&self, values: &[f64]) -> Result<()> {
        if values.len() != self.num_slots {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "expected {} parameter values, got {}",
                    self.num_slots,
                    values.len()
                ),
            });
        }
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "parameter value {i} is {}, expected a finite angle",
                    values[i]
                ),
            });
        }
        Ok(())
    }

    /// Overwrite the linked angles of a circuit already shaped like the
    /// template. Callers must have validated the links against it.
    pub(crate) fn write_angles(&self, out: &mut Circuit, values: &[f64]) {
        for link in &self.links {
            *angle_mut(&mut out.instructions[link.instruction]) = values[link.slot];
        }
    }

    /// Read the angle each slot currently holds in `circuit`.
    ///
    /// # Errors
    /// Same conditions as [`validate`](Self::validate), which this runs first
    /// because a link pointing at a non-angle gate has no value to read.
    pub fn values(&self, circuit: &Circuit) -> Result<Vec<f64>> {
        self.validate(circuit)?;
        let mut out = vec![0.0; self.num_slots];
        for link in &self.links {
            out[link.slot] = angle_of(&circuit.instructions[link.instruction]);
        }
        Ok(out)
    }
}

/// Angle of a gate already validated as bindable.
pub(crate) fn angle_of(instruction: &Instruction) -> f64 {
    match instruction {
        Instruction::Gate {
            gate: Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::Rzz(t) | Gate::P(t),
            ..
        } => *t,
        Instruction::Gate {
            gate: Gate::PauliRot(data),
            ..
        } => data.theta(),
        _ => unreachable!("parameter link validated as bindable"),
    }
}

pub(crate) fn angle_mut(instruction: &mut Instruction) -> &mut f64 {
    match instruction {
        Instruction::Gate {
            gate: Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::Rzz(t) | Gate::P(t),
            ..
        } => t,
        Instruction::Gate {
            gate: Gate::PauliRot(data),
            ..
        } => &mut data.theta,
        _ => unreachable!("parameter link validated as bindable"),
    }
}

#[cfg(test)]
#[path = "parameter_tests.rs"]
mod tests;
