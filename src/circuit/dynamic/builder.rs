//! Structured construction of a [`DynamicProgram`].

use super::{
    Action, BasicBlock, BlockId, ClassicalExpr, ClassicalType, ClassicalValue, DynamicProgram,
    RotationKind, Terminator, VarId, Variable,
};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

struct PendingBlock {
    instructions: Vec<Instruction>,
    actions: Vec<Action>,
    terminator: Option<Terminator>,
    loop_id: Option<usize>,
}

enum Frame {
    While {
        loop_id: usize,
        header: usize,
        exit: usize,
    },
    If {
        branch: usize,
        join: usize,
        in_else: bool,
    },
}

/// Builds a [`DynamicProgram`] from structured control flow.
///
/// Instructions append to the block being built. [`begin_while`] and
/// [`begin_if`] open a construct that [`end`] closes; [`begin_else`] switches
/// the innermost `if` to its other arm, or reopens an `if` that [`end`] has
/// just closed. Every index is checked when [`build`] validates the program,
/// so no method here panics on a bad one.
///
/// # Examples
///
/// Repeat until success: prepare `|+>`, measure, and retry while the outcome
/// is 1.
///
/// ```
/// use prism_q::circuit::dynamic::{ClassicalExpr, DynamicProgramBuilder};
/// use prism_q::{Gate, simulate_program};
///
/// let mut b = DynamicProgramBuilder::new(1, 1);
/// b.add_gate(Gate::H, &[0]).add_measure(0, 0);
/// b.begin_while("retry", ClassicalExpr::Bit(0));
/// b.add_reset(0).add_gate(Gate::H, &[0]).add_measure(0, 0);
/// b.end()?;
/// let program = b.build()?;
///
/// let shots = simulate_program(&program).seed(42).shots(100)?;
/// assert!(shots.shots.iter().all(|shot| !shot[0]));
/// # Ok::<(), prism_q::PrismError>(())
/// ```
///
/// [`begin_while`]: Self::begin_while
/// [`begin_if`]: Self::begin_if
/// [`begin_else`]: Self::begin_else
/// [`end`]: Self::end
/// [`build`]: Self::build
pub struct DynamicProgramBuilder {
    num_qubits: usize,
    num_classical_bits: usize,
    variables: Vec<Variable>,
    blocks: Vec<PendingBlock>,
    current: usize,
    frames: Vec<Frame>,
    loop_names: Vec<String>,
    /// The `if` frame [`Self::end`] just closed, which [`Self::begin_else`] may
    /// reopen until anything else is appended.
    reopenable: Option<Frame>,
}

impl DynamicProgramBuilder {
    pub fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self {
            num_qubits,
            num_classical_bits,
            variables: Vec::new(),
            blocks: vec![PendingBlock {
                instructions: Vec::new(),
                actions: Vec::new(),
                terminator: None,
                loop_id: None,
            }],
            current: 0,
            frames: Vec::new(),
            loop_names: Vec::new(),
            reopenable: None,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn num_classical_bits(&self) -> usize {
        self.num_classical_bits
    }

    pub(crate) fn resize(&mut self, num_qubits: usize, num_classical_bits: usize) {
        self.num_qubits = num_qubits;
        self.num_classical_bits = num_classical_bits;
    }

    /// Declare a variable every shot starts at `initial`.
    pub fn declare(
        &mut self,
        name: impl Into<String>,
        ty: ClassicalType,
        initial: ClassicalValue,
    ) -> VarId {
        self.variables.push(Variable {
            name: name.into(),
            ty,
            initial,
        });
        VarId::new(self.variables.len() - 1)
    }

    /// The variable declared under `name`, the latest one when several share it.
    pub fn variable(&self, name: &str) -> Option<VarId> {
        self.variables
            .iter()
            .rposition(|variable| variable.name == name)
            .map(VarId::new)
    }

    /// Parse an OpenQASM expression over this builder's variables, by name, and
    /// its classical bits, read as `c[i]` for one bit and `c` for all of them as
    /// an unsigned integer.
    ///
    /// # Errors
    ///
    /// [`PrismError::Parse`] for text that is not an expression, an unknown
    /// name, or a builtin function over a runtime value.
    pub fn expr(&self, text: &str) -> Result<ClassicalExpr> {
        crate::circuit::openqasm::builder_expr(text, self)
    }

    pub fn add_instruction(&mut self, instruction: Instruction) -> &mut Self {
        let at = self.circuit_block();
        self.blocks[at].instructions.push(instruction);
        self
    }

    pub fn add_gate(&mut self, gate: Gate, targets: &[usize]) -> &mut Self {
        self.add_instruction(Instruction::Gate {
            gate,
            targets: SmallVec::from_slice(targets),
        })
    }

    pub fn add_measure(&mut self, qubit: usize, classical_bit: usize) -> &mut Self {
        self.add_instruction(Instruction::Measure {
            qubit,
            classical_bit,
        })
    }

    pub fn add_reset(&mut self, qubit: usize) -> &mut Self {
        self.add_instruction(Instruction::Reset { qubit })
    }

    /// Append every instruction of `circuit`.
    pub fn append(&mut self, circuit: &Circuit) -> &mut Self {
        let at = self.circuit_block();
        self.blocks[at]
            .instructions
            .extend(circuit.instructions.iter().cloned());
        self
    }

    /// Apply `kind` to `targets` at the angle `angle` evaluates to when the
    /// block runs.
    pub fn add_rotation(
        &mut self,
        kind: RotationKind,
        targets: &[usize],
        angle: ClassicalExpr,
    ) -> &mut Self {
        self.push_action(Action::Rotation {
            kind,
            targets: SmallVec::from_slice(targets),
            angle,
        })
    }

    /// Store `value` into `var`, converted to its declared type.
    pub fn assign(&mut self, var: VarId, value: ClassicalExpr) -> &mut Self {
        self.push_action(Action::Assign { var, value })
    }

    /// Open a loop that runs its body while `condition` is truthy, testing it
    /// before each pass. `name` labels the loop in the step-limit error.
    pub fn begin_while(&mut self, name: impl Into<String>, condition: ClassicalExpr) {
        self.reopenable = None;
        let enclosing = self.blocks[self.current].loop_id;
        let loop_id = self.loop_names.len();
        self.loop_names.push(name.into());
        let header = self.new_block(Some(loop_id));
        let body = self.new_block(Some(loop_id));
        let exit = self.new_block(enclosing);
        self.terminate(Terminator::Jump(BlockId::new(header)));
        self.blocks[header].terminator = Some(Terminator::Branch {
            condition,
            then: BlockId::new(body),
            otherwise: BlockId::new(exit),
        });
        self.frames.push(Frame::While {
            loop_id,
            header,
            exit,
        });
        self.current = body;
    }

    /// Open a branch whose body runs when `condition` is truthy.
    pub fn begin_if(&mut self, condition: ClassicalExpr) {
        self.reopenable = None;
        let loop_id = self.blocks[self.current].loop_id;
        let then = self.new_block(loop_id);
        let join = self.new_block(loop_id);
        self.terminate(Terminator::Branch {
            condition,
            then: BlockId::new(then),
            otherwise: BlockId::new(join),
        });
        self.frames.push(Frame::If {
            branch: self.current,
            join,
            in_else: false,
        });
        self.current = then;
    }

    /// Switch the innermost open `if` to its `else` arm, or reopen the `if`
    /// that [`Self::end`] closed last when nothing has been appended since.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidParameter`] when there is no such `if`, or it
    /// already has an `else`.
    pub fn begin_else(&mut self) -> Result<()> {
        let open = matches!(self.frames.last(), Some(Frame::If { in_else: false, .. }));
        let (branch, join) = if open {
            let Some(Frame::If { branch, join, .. }) = self.frames.pop() else {
                unreachable!("checked above")
            };
            self.terminate(Terminator::Jump(BlockId::new(join)));
            (branch, join)
        } else {
            match self.reopenable.take() {
                Some(Frame::If {
                    branch,
                    join,
                    in_else: false,
                }) => (branch, join),
                _ => return Err(misuse("`begin_else` with no `if` to attach to")),
            }
        };
        let other = self.new_block(self.blocks[join].loop_id);
        if let Some(Terminator::Branch { otherwise, .. }) = &mut self.blocks[branch].terminator {
            *otherwise = BlockId::new(other);
        }
        self.frames.push(Frame::If {
            branch,
            join,
            in_else: true,
        });
        self.current = other;
        Ok(())
    }

    /// Close the innermost `while` or `if`.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidParameter`] when nothing is open.
    pub fn end(&mut self) -> Result<()> {
        let frame = self
            .frames
            .pop()
            .ok_or_else(|| misuse("`end` with no open `while` or `if`"))?;
        match frame {
            Frame::While { header, exit, .. } => {
                self.terminate(Terminator::Jump(BlockId::new(header)));
                self.current = exit;
                self.reopenable = None;
            }
            Frame::If {
                branch,
                join,
                in_else,
            } => {
                self.terminate(Terminator::Jump(BlockId::new(join)));
                self.current = join;
                self.reopenable = Some(Frame::If {
                    branch,
                    join,
                    in_else,
                });
            }
        }
        Ok(())
    }

    /// Leave the innermost `while`.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidParameter`] outside a loop.
    pub fn break_loop(&mut self) -> Result<()> {
        let exit = self
            .innermost_loop()
            .map(|(_, exit)| exit)
            .ok_or_else(|| misuse("`break` outside a `while` loop"))?;
        self.jump_away(exit);
        Ok(())
    }

    /// Skip to the next test of the innermost `while` condition.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidParameter`] outside a loop.
    pub fn continue_loop(&mut self) -> Result<()> {
        let header = self
            .innermost_loop()
            .map(|(header, _)| header)
            .ok_or_else(|| misuse("`continue` outside a `while` loop"))?;
        self.jump_away(header);
        Ok(())
    }

    /// Finish the program, dropping unreachable blocks and merging each
    /// straight-line run of blocks into one, so fusion sees the longest
    /// circuits the control flow allows.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidParameter`] while a `while` or `if` is still open,
    /// and anything [`DynamicProgram::new`] rejects.
    pub fn build(mut self) -> Result<DynamicProgram> {
        if let Some(frame) = self.frames.last() {
            let what = match frame {
                Frame::While { loop_id, .. } => {
                    format!("`while` loop `{}`", self.loop_names[*loop_id])
                }
                Frame::If { .. } => "`if`".to_string(),
            };
            return Err(misuse(format!("{what} is still open at `build`")));
        }
        for block in &mut self.blocks {
            block.terminator.get_or_insert(Terminator::End);
        }
        let blocks = compact(self.blocks);
        let blocks = blocks
            .into_iter()
            .map(|block| BasicBlock {
                circuit: Circuit {
                    num_qubits: self.num_qubits,
                    num_classical_bits: self.num_classical_bits,
                    instructions: block.instructions,
                },
                actions: block.actions,
                terminator: block.terminator.expect("every block terminated above"),
                loop_name: block.loop_id.map(|id| self.loop_names[id].clone()),
            })
            .collect();
        DynamicProgram::new(
            self.num_qubits,
            self.num_classical_bits,
            self.variables,
            blocks,
        )
    }

    fn new_block(&mut self, loop_id: Option<usize>) -> usize {
        self.blocks.push(PendingBlock {
            instructions: Vec::new(),
            actions: Vec::new(),
            terminator: None,
            loop_id,
        });
        self.blocks.len() - 1
    }

    fn terminate(&mut self, terminator: Terminator) {
        self.blocks[self.current]
            .terminator
            .get_or_insert(terminator);
    }

    /// The block instructions append to: the current one, unless it already
    /// holds an action, which a circuit may not follow within one block.
    fn circuit_block(&mut self) -> usize {
        self.reopenable = None;
        if !self.blocks[self.current].actions.is_empty() {
            let next = self.new_block(self.blocks[self.current].loop_id);
            self.terminate(Terminator::Jump(BlockId::new(next)));
            self.current = next;
        }
        self.current
    }

    fn push_action(&mut self, action: Action) -> &mut Self {
        self.reopenable = None;
        self.blocks[self.current].actions.push(action);
        self
    }

    fn innermost_loop(&self) -> Option<(usize, usize)> {
        self.frames.iter().rev().find_map(|frame| match frame {
            Frame::While { header, exit, .. } => Some((*header, *exit)),
            Frame::If { .. } => None,
        })
    }

    /// End the current block with a jump and continue in a fresh one, which
    /// nothing reaches until a later construct joins into it.
    fn jump_away(&mut self, target: usize) {
        self.reopenable = None;
        self.terminate(Terminator::Jump(BlockId::new(target)));
        self.current = self.new_block(self.blocks[self.current].loop_id);
    }
}

fn misuse(message: impl Into<String>) -> PrismError {
    PrismError::InvalidParameter {
        message: message.into(),
    }
}

/// Thread jumps through empty blocks, merge every block into the one block
/// that jumps to it alone, then drop what the entry cannot reach and renumber
/// in reachable order.
fn compact(mut blocks: Vec<PendingBlock>) -> Vec<PendingBlock> {
    let forward = |blocks: &[PendingBlock], start: usize| {
        let mut at = start;
        for _ in 0..blocks.len() {
            let block = &blocks[at];
            match block.terminator {
                Some(Terminator::Jump(next))
                    if block.instructions.is_empty()
                        && block.actions.is_empty()
                        && next.index() != at =>
                {
                    at = next.index();
                }
                _ => break,
            }
        }
        at
    };
    let threaded: Vec<usize> = (0..blocks.len()).map(|at| forward(&blocks, at)).collect();
    for block in &mut blocks {
        retarget(block, |id| BlockId::new(threaded[id.index()]));
    }
    let entry = threaded[0];

    let mut predecessors = vec![0usize; blocks.len()];
    for block in &blocks {
        for target in successors(block) {
            predecessors[target] += 1;
        }
    }
    for at in 0..blocks.len() {
        while let Some(Terminator::Jump(target)) = blocks[at].terminator {
            let from = target.index();
            if !blocks[at].actions.is_empty()
                || from == at
                || from == entry
                || predecessors[from] != 1
            {
                break;
            }
            let absorbed = std::mem::replace(
                &mut blocks[from],
                PendingBlock {
                    instructions: Vec::new(),
                    actions: Vec::new(),
                    terminator: Some(Terminator::End),
                    loop_id: None,
                },
            );
            let block = &mut blocks[at];
            block.instructions.extend(absorbed.instructions);
            block.actions = absorbed.actions;
            block.terminator = absorbed.terminator;
        }
    }

    let mut order = Vec::with_capacity(blocks.len());
    let mut renumber = vec![usize::MAX; blocks.len()];
    let mut stack = vec![entry];
    while let Some(at) = stack.pop() {
        if renumber[at] != usize::MAX {
            continue;
        }
        renumber[at] = order.len();
        order.push(at);
        for target in successors(&blocks[at]).into_iter().rev() {
            if renumber[target] == usize::MAX {
                stack.push(target);
            }
        }
    }
    let mut slots: Vec<Option<PendingBlock>> = blocks.into_iter().map(Some).collect();
    order
        .iter()
        .map(|&at| {
            let mut block = slots[at].take().expect("each block visited once");
            retarget(&mut block, |id| BlockId::new(renumber[id.index()]));
            block
        })
        .collect()
}

fn successors(block: &PendingBlock) -> SmallVec<[usize; 2]> {
    match block.terminator.as_ref().expect("every block terminated") {
        Terminator::Jump(target) => SmallVec::from_slice(&[target.index()]),
        Terminator::Branch {
            then, otherwise, ..
        } => SmallVec::from_slice(&[then.index(), otherwise.index()]),
        Terminator::End => SmallVec::new(),
    }
}

fn retarget(block: &mut PendingBlock, map: impl Fn(BlockId) -> BlockId) {
    match block.terminator.as_mut().expect("every block terminated") {
        Terminator::Jump(target) => *target = map(*target),
        Terminator::Branch {
            then, otherwise, ..
        } => {
            *then = map(*then);
            *otherwise = map(*otherwise);
        }
        Terminator::End => {}
    }
}
