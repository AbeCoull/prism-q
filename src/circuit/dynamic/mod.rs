//! Programs with runtime control flow: basic blocks of ordinary circuits joined
//! by branches on classical expressions, run once per shot.
//!
//! [`openqasm::parse_dynamic`](super::openqasm::parse_dynamic) builds one from
//! source, [`DynamicProgramBuilder`] builds one directly, and
//! [`simulate_program`](crate::sim::simulate_program) runs it.

mod builder;
mod expr;

pub use builder::DynamicProgramBuilder;
pub(crate) use expr::check_bit;
pub use expr::{BinaryOp, ClassicalExpr, ClassicalType, ClassicalValue, UnaryOp, VarId};

use super::{Circuit, ClassicalCondition, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

/// Index of a basic block in a [`DynamicProgram`]. Block 0 is the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(u32);

impl BlockId {
    /// # Panics
    ///
    /// Panics when `index` does not fit in 32 bits.
    pub fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("block index fits in 32 bits"))
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A runtime classical variable. Every shot starts it at `initial`, stored
/// through `ty`.
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub ty: ClassicalType,
    pub initial: ClassicalValue,
}

/// Rotation whose angle a dynamic program computes at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RotationKind {
    Rx,
    Ry,
    Rz,
    /// The phase gate `P(theta)`.
    Phase,
    /// `exp(-i theta Z⊗Z / 2)` on two qubits.
    Rzz,
}

impl RotationKind {
    pub(crate) fn gate(self, angle: f64) -> Gate {
        match self {
            RotationKind::Rx => Gate::Rx(angle),
            RotationKind::Ry => Gate::Ry(angle),
            RotationKind::Rz => Gate::Rz(angle),
            RotationKind::Phase => Gate::P(angle),
            RotationKind::Rzz => Gate::Rzz(angle),
        }
    }

    fn arity(self) -> usize {
        match self {
            RotationKind::Rzz => 2,
            _ => 1,
        }
    }
}

/// A step a block takes after its circuit, in order.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Action {
    /// Store `value` into `var`, converted to the variable's type.
    Assign { var: VarId, value: ClassicalExpr },
    /// Apply `kind` at the angle `angle` evaluates to, in radians.
    Rotation {
        kind: RotationKind,
        targets: SmallVec<[usize; 4]>,
        angle: ClassicalExpr,
    },
}

/// Where control goes after a block.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Terminator {
    Jump(BlockId),
    /// Go to `then` when `condition` is truthy (a nonzero number or `true`),
    /// to `otherwise` when it is not.
    Branch {
        condition: ClassicalExpr,
        then: BlockId,
        otherwise: BlockId,
    },
    End,
}

/// One basic block: its circuit runs first, then its actions, then its
/// terminator picks the next block.
///
/// `circuit` is an ordinary [`Circuit`] over the program's full width, and may
/// carry guarded regions on classical bits. Fusion runs on it once, when the
/// program is prepared for a backend.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub circuit: Circuit,
    pub actions: Vec<Action>,
    pub terminator: Terminator,
    /// The loop this block belongs to, which the step-limit error names.
    pub loop_name: Option<String>,
}

/// A program whose instruction count can depend on measurement outcomes: a
/// control-flow graph of [`BasicBlock`]s plus runtime classical variables.
///
/// Construction validates the graph: every block target, variable, qubit, and
/// classical bit a block names exists, and no block carries a save point.
#[derive(Debug, Clone)]
pub struct DynamicProgram {
    num_qubits: usize,
    num_classical_bits: usize,
    variables: Vec<Variable>,
    blocks: Vec<BasicBlock>,
}

impl DynamicProgram {
    /// Validate and assemble a program. Each variable's `initial` value is
    /// stored through its type, so an out-of-range integer wraps here.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidQubit`] and [`PrismError::InvalidClassicalBit`] for
    /// an index past the program's width, and [`PrismError::InvalidParameter`]
    /// for anything else malformed: no blocks, a block circuit of the wrong
    /// width, a missing block or variable, a save point, a bitwise operator on
    /// a float, or a rotation with the wrong number of targets.
    pub fn new(
        num_qubits: usize,
        num_classical_bits: usize,
        mut variables: Vec<Variable>,
        blocks: Vec<BasicBlock>,
    ) -> Result<Self> {
        if blocks.is_empty() {
            return Err(invalid("a dynamic program needs at least one block"));
        }
        for variable in &mut variables {
            variable.ty.check()?;
            variable.initial = variable.ty.store(variable.initial);
        }
        let types: Vec<ClassicalType> = variables.iter().map(|variable| variable.ty).collect();
        let check_target = |id: BlockId| {
            if id.index() >= blocks.len() {
                return Err(invalid(format!(
                    "a terminator names block {} of {}",
                    id.index(),
                    blocks.len()
                )));
            }
            Ok(())
        };
        for (at, block) in blocks.iter().enumerate() {
            let circuit = &block.circuit;
            if circuit.num_qubits != num_qubits || circuit.num_classical_bits != num_classical_bits
            {
                return Err(invalid(format!(
                    "block {at} holds a circuit of {} qubits and {} bits in a program of {} and {}",
                    circuit.num_qubits, circuit.num_classical_bits, num_qubits, num_classical_bits
                )));
            }
            check_instructions(&circuit.instructions, num_qubits, num_classical_bits)?;
            for action in &block.actions {
                match action {
                    Action::Assign { var, value } => {
                        if var.index() >= variables.len() {
                            return Err(invalid(format!(
                                "block {at} assigns variable {} of {} declared",
                                var.index(),
                                variables.len()
                            )));
                        }
                        value.check(&types, num_classical_bits)?;
                    }
                    Action::Rotation {
                        kind,
                        targets,
                        angle,
                    } => {
                        if targets.len() != kind.arity() {
                            return Err(PrismError::GateArity {
                                gate: format!("{kind:?}").to_lowercase(),
                                expected: kind.arity(),
                                got: targets.len(),
                            });
                        }
                        check_qubits(targets, num_qubits)?;
                        if targets.len() == 2 && targets[0] == targets[1] {
                            return Err(invalid(format!(
                                "block {at} applies a two-qubit rotation to qubit {} twice",
                                targets[0]
                            )));
                        }
                        angle.check(&types, num_classical_bits)?;
                    }
                }
            }
            match &block.terminator {
                Terminator::Jump(target) => check_target(*target)?,
                Terminator::Branch {
                    condition,
                    then,
                    otherwise,
                } => {
                    condition.check(&types, num_classical_bits)?;
                    check_target(*then)?;
                    check_target(*otherwise)?;
                }
                Terminator::End => {}
            }
        }
        Ok(Self {
            num_qubits,
            num_classical_bits,
            variables,
            blocks,
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn num_classical_bits(&self) -> usize {
        self.num_classical_bits
    }

    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// Blocks in index order; block 0 is the entry.
    pub fn blocks(&self) -> &[BasicBlock] {
        &self.blocks
    }

    /// The program's circuit when it has no runtime control flow: one block,
    /// no actions, and nothing after it. Such a program runs exactly as that
    /// circuit does, through every sampling shortcut the circuit qualifies for.
    pub fn static_circuit(&self) -> Option<&Circuit> {
        match self.blocks.as_slice() {
            [block] if block.actions.is_empty() && block.terminator == Terminator::End => {
                Some(&block.circuit)
            }
            _ => None,
        }
    }

    /// Every block's instructions in one circuit, with each runtime rotation
    /// standing in at a generic angle, for the routing decisions that read a
    /// circuit's gate set.
    pub(crate) fn summary_circuit(&self) -> Circuit {
        let mut summary = Circuit::new(self.num_qubits, self.num_classical_bits);
        for block in &self.blocks {
            summary
                .instructions
                .extend(block.circuit.instructions.iter().cloned());
            for action in &block.actions {
                if let Action::Rotation { kind, targets, .. } = action {
                    summary.instructions.push(Instruction::Gate {
                        gate: kind.gate(1.0),
                        targets: targets.clone(),
                    });
                }
            }
        }
        summary
    }
}

fn invalid(message: impl Into<String>) -> PrismError {
    PrismError::InvalidParameter {
        message: message.into(),
    }
}

fn check_qubits(qubits: &[usize], num_qubits: usize) -> Result<()> {
    match qubits.iter().find(|&&qubit| qubit >= num_qubits) {
        Some(&index) => Err(PrismError::InvalidQubit {
            index,
            register_size: num_qubits,
        }),
        None => Ok(()),
    }
}

fn check_condition(condition: &ClassicalCondition, num_classical_bits: usize) -> Result<()> {
    match condition {
        ClassicalCondition::BitIsOne(bit) | ClassicalCondition::BitIsZero(bit) => {
            check_bit(*bit, num_classical_bits)
        }
        ClassicalCondition::RegisterEquals { offset, size, .. }
        | ClassicalCondition::RegisterNotEquals { offset, size, .. } => match size {
            0 => Ok(()),
            _ => check_bit(offset + size - 1, num_classical_bits),
        },
        ClassicalCondition::Parity { bits, .. } => bits
            .iter()
            .try_for_each(|&bit| check_bit(bit, num_classical_bits)),
    }
}

fn check_instructions(
    instructions: &[Instruction],
    num_qubits: usize,
    num_classical_bits: usize,
) -> Result<()> {
    for instruction in instructions {
        match instruction {
            Instruction::Gate { gate, targets } => {
                if gate.num_qubits() != targets.len() {
                    return Err(PrismError::GateArity {
                        gate: gate.name().to_string(),
                        expected: gate.num_qubits(),
                        got: targets.len(),
                    });
                }
                check_qubits(targets, num_qubits)?;
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                check_qubits(&[*qubit], num_qubits)?;
                check_bit(*classical_bit, num_classical_bits)?;
            }
            Instruction::Reset { qubit } => check_qubits(&[*qubit], num_qubits)?,
            Instruction::Barrier { qubits } => check_qubits(qubits, num_qubits)?,
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                check_condition(condition, num_classical_bits)?;
                if gate.num_qubits() != targets.len() {
                    return Err(PrismError::GateArity {
                        gate: gate.name().to_string(),
                        expected: gate.num_qubits(),
                        got: targets.len(),
                    });
                }
                check_qubits(targets, num_qubits)?;
            }
            Instruction::Region(region) => {
                check_condition(region.condition(), num_classical_bits)?;
                check_instructions(region.body(), num_qubits, num_classical_bits)?;
            }
            Instruction::Save { label, .. } => {
                return Err(invalid(format!(
                    "save point `{label}` in a dynamic program, whose shots have nowhere to \
                     return it"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
