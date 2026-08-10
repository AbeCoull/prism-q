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
/// Bindable gates are `Rx`, `Ry`, `Rz`, `Rzz`, and `P`, the `Gate` variants
/// carrying an angle inline.
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

    /// Slot a name refers to.
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

    /// Give every bindable gate its own slot, in circuit order. The common
    /// case for a variational ansatz where each rotation is independent.
    pub fn all_rotations(circuit: &Circuit) -> Self {
        let mut links = Vec::new();
        for (i, inst) in circuit.instructions.iter().enumerate() {
            if let Instruction::Gate { gate, .. } = inst {
                if gate.pauli_generator().is_some() {
                    links.push(ParamLink {
                        instruction: i,
                        slot: links.len(),
                    });
                }
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
    /// Panics if `slot` is not below the declared slot count. Slot bounds are
    /// fixed when the set is constructed, so an out-of-range slot is a caller
    /// bug rather than bad input.
    pub fn link(&mut self, instruction: usize, slot: usize) {
        assert!(
            slot < self.num_slots,
            "slot {} out of bounds (parameter set declares {} slots)",
            slot,
            self.num_slots
        );
        self.links.push(ParamLink { instruction, slot });
    }

    /// The recorded links.
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
    /// carrying no angle.
    pub fn validate(&self, circuit: &Circuit) -> Result<()> {
        let n = circuit.instructions.len();
        for link in &self.links {
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
                            "instruction {} (`{}`) carries no bindable angle; bindable gates are rx, ry, rz, rzz, p",
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
fn angle_of(instruction: &Instruction) -> f64 {
    match instruction {
        Instruction::Gate {
            gate: Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::Rzz(t) | Gate::P(t),
            ..
        } => *t,
        _ => unreachable!("parameter link validated as bindable"),
    }
}

pub(crate) fn angle_mut(instruction: &mut Instruction) -> &mut f64 {
    match instruction {
        Instruction::Gate {
            gate: Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::Rzz(t) | Gate::P(t),
            ..
        } => t,
        _ => unreachable!("parameter link validated as bindable"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_rotations() -> Circuit {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Rx(0.1), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(0.2), &[1]);
        c
    }

    #[test]
    fn all_rotations_declares_one_slot_per_gate() {
        let p = Parameters::all_rotations(&two_rotations());
        assert_eq!(p.num_slots(), 2);
        assert_eq!(p.links().len(), 2);
        assert_eq!(p.links()[1].instruction, 2);
    }

    #[test]
    fn bind_writes_angles_and_leaves_structure() {
        let template = two_rotations();
        let p = Parameters::all_rotations(&template);
        let bound = p.bind(&template, &[1.5, 2.5]).unwrap();
        assert_eq!(bound.instructions.len(), 3);
        assert!(matches!(
            bound.instructions[0],
            Instruction::Gate {
                gate: Gate::Rx(t),
                ..
            } if t == 1.5
        ));
        assert!(matches!(
            bound.instructions[2],
            Instruction::Gate {
                gate: Gate::Rz(t),
                ..
            } if t == 2.5
        ));
    }

    #[test]
    fn shared_slot_writes_every_linked_gate() {
        let template = two_rotations();
        let mut p = Parameters::new(1);
        p.link(0, 0);
        p.link(2, 0);
        let bound = p.bind(&template, &[0.75]).unwrap();
        assert_eq!(super::angle_of(&bound.instructions[0]), 0.75);
        assert_eq!(super::angle_of(&bound.instructions[2]), 0.75);
    }

    #[test]
    fn wrong_arity_is_an_error() {
        let template = two_rotations();
        let p = Parameters::all_rotations(&template);
        assert!(p.bind(&template, &[1.0]).is_err());
        assert!(p.bind(&template, &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn non_finite_value_is_an_error() {
        let template = two_rotations();
        let p = Parameters::all_rotations(&template);
        assert!(p.bind(&template, &[f64::NAN, 0.0]).is_err());
        assert!(p.bind(&template, &[0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn link_past_end_is_an_error() {
        let template = two_rotations();
        let mut p = Parameters::new(1);
        p.link(99, 0);
        assert!(p.bind(&template, &[0.5]).is_err());
    }

    #[test]
    fn link_to_non_bindable_gate_is_an_error() {
        let template = two_rotations();
        let mut p = Parameters::new(1);
        p.link(1, 0);
        assert!(p.bind(&template, &[0.5]).is_err());
    }

    #[test]
    fn slot_no_gate_reads_is_accepted_and_reported() {
        let template = two_rotations();
        let mut p = Parameters::new(2);
        p.link(0, 0);
        assert!(p.bind(&template, &[0.5, 0.5]).is_ok());
        assert_eq!(p.unread_slots(), vec![1]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slot_past_declared_count_panics() {
        let mut p = Parameters::new(1);
        p.link(0, 4);
    }

    #[test]
    fn values_round_trip_through_bind() {
        let template = two_rotations();
        let p = Parameters::all_rotations(&template);
        let bound = p.bind(&template, &[0.3, 0.4]).unwrap();
        assert_eq!(p.values(&bound).unwrap(), vec![0.3, 0.4]);
    }
}
