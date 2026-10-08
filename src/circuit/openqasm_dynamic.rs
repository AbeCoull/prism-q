//! Lowering a program with runtime control flow into a [`DynamicProgram`].
//!
//! A pass over the tree first decides which classical variables live at
//! runtime: one written under a loop or a measured condition its declaration
//! sits outside, or assigned from a measured bit or another runtime variable.
//! Every other variable folds at parse time exactly as [`parse`] folds it, so a
//! program with no runtime control flow lowers to the circuit `parse` returns.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use super::*;
use crate::circuit::dynamic::{
    self as dynamic, ClassicalExpr, ClassicalType as RuntimeType, ClassicalValue, DynamicProgram,
    DynamicProgramBuilder, RotationKind, UnaryOp, VarId,
};
use crate::circuit::qasm::ast::{AssignOp, Block, Condition, Operand, Stmt, StmtKind};
use crate::circuit::qasm::expr::{self as syntax_expr, BinaryOp, Expr};
use crate::circuit::qasm::stream::Stream;
use crate::circuit::qasm::{lexer, parser as syntax};

/// What the evaluator carries while lowering into a control-flow graph.
pub(super) struct DynamicState<'a> {
    builder: DynamicProgramBuilder,
    analysis: RuntimeAnalysis,
    /// Runtime variables in scope, by name.
    vars: HashMap<&'a str, VarId>,
    /// Nesting of guarded regions being collected into one instruction, where
    /// nothing may open a block.
    collecting: usize,
    /// Enclosing loops, innermost last: `true` for a `while`, `false` for a
    /// `for` unrolled at parse time.
    loops: Vec<bool>,
    /// Constructs a declaration may run under other than once per shot: loop
    /// bodies and branches decided at runtime.
    repeat: usize,
}

/// Statement keys: the address of the statement in the tree, which every pass
/// over one body sees unchanged.
fn key(stmt: &Stmt) -> usize {
    stmt as *const Stmt as usize
}

/// Parse an OpenQASM 3.0 program that may need runtime control flow into a
/// [`DynamicProgram`].
///
/// Reads everything [`parse`] reads, plus `while` loops with `break` and
/// `continue`, conditions and assignments over the full classical operator set
/// (`&&`, `||`, `!`, comparisons, `&`, `|`, `^`, `~`, shifts), classical bits
/// and registers read as values, writes to a variable under a measured `if`,
/// and a runtime value as the angle of `rx`, `ry`, `rz`, `p` or `rzz`. A
/// variable written only where its value is known at parse time still folds
/// there, so a program `parse` accepts lowers to one block holding the same
/// circuit.
///
/// Declined, each with [`PrismError::UnsupportedConstruct`]: `input`
/// parameters, a register or array whose contents a loop or measurement
/// decides, a runtime value where a parse-time constant belongs (an index, a
/// loop bound, a gate angle outside the five above, a `def` argument), a
/// builtin function over a runtime value, `break` or `continue` inside a `for`,
/// and a measurement into a variable rather than a bit.
///
/// # Errors
///
/// Returns [`PrismError`] for any parse failure or unsupported construct.
///
/// # Examples
///
/// ```
/// use prism_q::circuit::openqasm;
/// use prism_q::simulate_program;
///
/// let program = openqasm::parse_dynamic(
///     "OPENQASM 3.0;
///      qubit[1] q;
///      bit[1] c;
///      h q[0];
///      c[0] = measure q[0];
///      while (c[0]) { reset q[0]; h q[0]; c[0] = measure q[0]; }",
/// )?;
/// let shots = simulate_program(&program).seed(42).shots(100)?;
/// assert!(shots.shots.iter().all(|shot| !shot[0]));
/// # Ok::<(), prism_q::PrismError>(())
/// ```
pub fn parse_dynamic(input: &str) -> Result<DynamicProgram> {
    let tokens = lexer::tokenize(input)?;
    let program = syntax::parse_dynamic_program(&tokens)?;
    let mut parser = Parser::new_with(input, Dialect::Native);
    parser.dynamic = Some(Box::new(DynamicState {
        builder: DynamicProgramBuilder::new(0, 0),
        analysis: RuntimeAnalysis::of(&program),
        vars: HashMap::new(),
        collecting: 0,
        loops: Vec::new(),
        repeat: 0,
    }));
    if let Some(highest) = super::eval::highest_physical(&program) {
        parser.physical = true;
        parser.total_qubits = highest + 1;
    }
    let tail = parser.execute(&program)?;
    parser.flush(tail);
    let state = parser.dynamic.take().expect("installed above");
    let mut builder = state.builder;
    builder.resize(parser.total_qubits, parser.total_cbits);
    builder.build()
}

/// Lower `text` against a builder's variables and its bits, which read as the
/// register `c`.
pub(crate) fn builder_expr(text: &str, builder: &DynamicProgramBuilder) -> Result<ClassicalExpr> {
    let tokens = lexer::tokenize(text)?;
    let mut stream = Stream::dynamic(&tokens);
    let expr = syntax_expr::parse(&mut stream)?;
    if !stream.at_end() {
        return Err(stream.expected("the end of the expression"));
    }
    let bits = builder.num_classical_bits();
    lower(
        &expr,
        1,
        &|leaf: &Expr| {
            Ok(match leaf {
                Expr::Ident(name) => match builder.variable(name) {
                    Some(var) => Some(ClassicalExpr::Var(var)),
                    None if *name == "c" => Some(register_read(0, bits, 1)?),
                    None => None,
                },
                Expr::Element(element) if element.array == "c" => {
                    let [index] = element.indices.as_slice() else {
                        return Err(parse_error(1, "`c` takes one index"));
                    };
                    let at = syntax_expr::eval(index, 1, None)?;
                    if at.fract() != 0.0 || at < 0.0 {
                        return Err(parse_error(1, format!("`c[{at}]` is not a bit")));
                    }
                    Some(ClassicalExpr::Bit(at as usize))
                }
                _ => None,
            })
        },
        &|folded: &Expr| syntax_expr::eval(folded, 1, None),
    )
}

/// `bits[offset..offset + size]` as an unsigned value, or one bit.
fn register_read(offset: usize, size: usize, line: usize) -> Result<ClassicalExpr> {
    match size {
        1 => Ok(ClassicalExpr::Bit(offset)),
        2..=64 => Ok(ClassicalExpr::Register { offset, size }),
        _ => Err(PrismError::UnsupportedConstruct {
            construct: format!("a {size}-bit register read as a value, past the 64 a value holds"),
            line,
        }),
    }
}

fn constant(value: f64) -> ClassicalExpr {
    ClassicalExpr::Const(if value.fract() == 0.0 && value.abs() < 9.0e18 {
        ClassicalValue::Int(value as i128)
    } else {
        ClassicalValue::Float(value)
    })
}

fn runtime_op(op: BinaryOp) -> dynamic::BinaryOp {
    use dynamic::BinaryOp as Op;
    match op {
        BinaryOp::Add => Op::Add,
        BinaryOp::Sub => Op::Sub,
        BinaryOp::Mul => Op::Mul,
        BinaryOp::Div => Op::Div,
        BinaryOp::Rem => Op::Rem,
        BinaryOp::Pow => Op::Pow,
        BinaryOp::Shl => Op::Shl,
        BinaryOp::Shr => Op::Shr,
        BinaryOp::BitAnd => Op::BitAnd,
        BinaryOp::BitOr => Op::BitOr,
        BinaryOp::BitXor => Op::BitXor,
        BinaryOp::Eq => Op::Eq,
        BinaryOp::Ne => Op::Ne,
        BinaryOp::Lt => Op::Lt,
        BinaryOp::Le => Op::Le,
        BinaryOp::Gt => Op::Gt,
        BinaryOp::Ge => Op::Ge,
        BinaryOp::And => Op::And,
        BinaryOp::Or => Op::Or,
    }
}

fn compound_op(op: AssignOp) -> dynamic::BinaryOp {
    use dynamic::BinaryOp as Op;
    match op {
        AssignOp::Add => Op::Add,
        AssignOp::Sub => Op::Sub,
        AssignOp::Mul => Op::Mul,
        AssignOp::Div => Op::Div,
        AssignOp::Rem => Op::Rem,
    }
}

/// Whether some leaf of `expr` is a runtime value according to `leaf`.
fn reads_leaf(expr: &Expr, leaf: &impl Fn(&Expr) -> Result<Option<ClassicalExpr>>) -> Result<bool> {
    Ok(match expr {
        Expr::Number(_) | Expr::Duration(_) => false,
        Expr::Ident(_) => leaf(expr)?.is_some(),
        Expr::Element(element) => {
            leaf(expr)?.is_some()
                || element.indices.iter().try_fold(false, |seen, index| {
                    Ok::<_, PrismError>(seen || reads_leaf(index, leaf)?)
                })?
        }
        Expr::Negate(inner) | Expr::Not(inner) | Expr::BitNot(inner) => reads_leaf(inner, leaf)?,
        Expr::Binary { left, right, .. } => reads_leaf(left, leaf)? || reads_leaf(right, leaf)?,
        Expr::Call(call) => call.args.iter().try_fold(false, |seen, arg| {
            Ok::<_, PrismError>(seen || reads_leaf(arg, leaf)?)
        })?,
    })
}

/// Lower `expr`, folding every subtree with no runtime leaf through `fold`.
fn lower(
    expr: &Expr,
    line: usize,
    leaf: &impl Fn(&Expr) -> Result<Option<ClassicalExpr>>,
    fold: &impl Fn(&Expr) -> Result<f64>,
) -> Result<ClassicalExpr> {
    if !reads_leaf(expr, leaf)? {
        return Ok(constant(fold(expr)?));
    }
    let recurse = |inner: &Expr| lower(inner, line, leaf, fold);
    Ok(match expr {
        Expr::Ident(_) | Expr::Element(_) => match leaf(expr)? {
            Some(lowered) => lowered,
            None => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`{expr}`, an array indexed by a runtime value"),
                    line,
                });
            }
        },
        Expr::Negate(inner) => ClassicalExpr::unary(UnaryOp::Neg, recurse(inner)?),
        Expr::Not(inner) => ClassicalExpr::unary(UnaryOp::Not, recurse(inner)?),
        Expr::BitNot(inner) => ClassicalExpr::unary(UnaryOp::BitNot, recurse(inner)?),
        Expr::Binary { op, left, right } => {
            ClassicalExpr::binary(runtime_op(*op), recurse(left)?, recurse(right)?)
        }
        Expr::Call(call) => {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "builtin `{}` over a runtime value; only operators run at runtime",
                    call.name
                ),
                line,
            });
        }
        Expr::Number(_) | Expr::Duration(_) => unreachable!("a literal reads no runtime leaf"),
    })
}

/// The expression a static condition form spells, for a condition over
/// variables rather than bits.
fn condition_expr<'a>(condition: &Condition<'a>) -> Result<Expr<'a>> {
    let binary = |op, left, right| Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    };
    let compare = |op: ast::CmpOp| match op {
        ast::CmpOp::Equal => BinaryOp::Eq,
        ast::CmpOp::NotEqual => BinaryOp::Ne,
    };
    Ok(match condition {
        Condition::Expr(expr) => expr.clone(),
        Condition::Truthy(operand) => operand_expr(operand)?,
        Condition::Negated(operand) => Expr::Not(Box::new(operand_expr(operand)?)),
        Condition::Compare { lhs, op, rhs } => {
            binary(compare(*op), operand_expr(lhs)?, rhs.clone())
        }
        Condition::Parity {
            bits,
            compare: test,
        } => {
            let mut parity = None;
            for bit in bits {
                let next = operand_expr(bit)?;
                parity = Some(match parity {
                    None => next,
                    Some(acc) => binary(BinaryOp::BitXor, acc, next),
                });
            }
            let parity = parity.expect("a parity names at least two bits");
            match test {
                None => binary(BinaryOp::Eq, parity, Expr::Number(1.0)),
                Some((op, rhs)) => binary(compare(*op), parity, rhs.clone()),
            }
        }
    })
}

/// An operand standing where a value belongs: a name, or one element of it.
fn operand_expr<'a>(operand: &Operand<'a>) -> Result<Expr<'a>> {
    let Some(name) = operand.register() else {
        return Err(parse_error(
            operand.line,
            format!("`{}` is a qubit where a value belongs", operand.describe()),
        ));
    };
    Ok(match &operand.index {
        None => Expr::Ident(name),
        Some(ast::Index::Single(index)) => Expr::Element(Box::new(syntax_expr::Element {
            array: name,
            indices: vec![index.clone()],
        })),
        Some(_) => {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("`{}`, a slice read as a runtime value", operand.describe()),
                line: operand.line,
            });
        }
    })
}

/// How a condition is decided.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Decided {
    /// At parse time: it reads neither a bit nor a runtime variable.
    Static,
    /// By a guard on measured bits, in one of the forms the instruction list
    /// has a condition for.
    Bits,
    /// By evaluating an expression when the program runs.
    Runtime,
}

impl<'a> Parser<'a> {
    pub(super) fn streaming(&self) -> bool {
        self.dynamic
            .as_ref()
            .is_some_and(|state| state.collecting == 0)
    }

    fn state(&mut self) -> &mut DynamicState<'a> {
        self.dynamic.as_mut().expect("dynamic lowering only")
    }

    /// Move instructions a statement produced into the block being built.
    pub(super) fn flush(&mut self, instructions: Vec<Instruction>) {
        if instructions.is_empty() {
            return;
        }
        let builder = &mut self.state().builder;
        for instruction in instructions {
            builder.add_instruction(instruction);
        }
    }

    fn require_streaming(&self, what: &str, line: usize) -> Result<()> {
        if self.streaming() {
            return Ok(());
        }
        Err(PrismError::UnsupportedConstruct {
            construct: format!(
                "{what} inside a guarded region, which lowers to one instruction; \
                 give the region a condition over a runtime value to make it a branch"
            ),
            line,
        })
    }

    pub(super) fn is_runtime_decl(&self, stmt: &Stmt) -> bool {
        self.dynamic
            .as_ref()
            .is_some_and(|state| state.analysis.runtime.contains(&key(stmt)))
    }

    pub(super) fn runtime_var(&self, name: &str) -> Option<VarId> {
        self.dynamic.as_ref()?.vars.get(name).copied()
    }

    pub(super) fn forget_runtime(&mut self, name: &str) {
        if let Some(state) = self.dynamic.as_mut() {
            state.vars.remove(name);
        }
    }

    pub(super) fn enter_collecting(&mut self) {
        if let Some(state) = self.dynamic.as_mut() {
            state.collecting += 1;
        }
    }

    pub(super) fn leave_collecting(&mut self) {
        if let Some(state) = self.dynamic.as_mut() {
            state.collecting -= 1;
        }
    }

    /// Enter a `for` body, which runs its declarations once per pass.
    pub(super) fn enter_static_loop(&mut self) {
        if let Some(state) = self.dynamic.as_mut() {
            state.loops.push(false);
            state.repeat += 1;
        }
    }

    pub(super) fn leave_static_loop(&mut self) {
        if let Some(state) = self.dynamic.as_mut() {
            state.loops.pop();
            state.repeat -= 1;
        }
    }

    pub(super) fn in_while(&self) -> bool {
        self.dynamic
            .as_ref()
            .is_some_and(|state| state.loops.contains(&true))
    }

    fn is_bit_name(&self, name: &str) -> bool {
        self.cregs.contains_key(name)
            || self
                .aliases
                .get(name)
                .is_some_and(|alias| alias.kind == ast::RegisterKind::Classical)
    }

    /// A classical bit or bits a name stands for, read as a value.
    fn bit_leaf(&self, name: &str, line: usize) -> Result<Option<ClassicalExpr>> {
        if let Some(register) = self.cregs.get(name) {
            return register_read(register.offset, register.size, line).map(Some);
        }
        match self.aliases.get(name) {
            Some(alias) if alias.kind == ast::RegisterKind::Classical => {
                let contiguous = alias.indices.windows(2).all(|pair| pair[1] == pair[0] + 1);
                if !contiguous {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "alias `{name}` read as a value; its bits do not run in order"
                        ),
                        line,
                    });
                }
                register_read(alias.indices[0], alias.indices.len(), line).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// The runtime value an identifier or an element stands for, `None` when
    /// it folds at parse time.
    fn runtime_leaf(&self, expr: &Expr, line: usize) -> Result<Option<ClassicalExpr>> {
        match expr {
            Expr::Ident(name) => {
                if let Some(var) = self.runtime_var(name) {
                    return Ok(Some(ClassicalExpr::Var(var)));
                }
                self.bit_leaf(name, line)
            }
            Expr::Element(element) => {
                if !self.is_bit_name(element.array) {
                    return Ok(None);
                }
                let operand = Operand {
                    name: ast::OperandName::Register(element.array),
                    index: match element.indices.as_slice() {
                        [index] => Some(ast::Index::Single(index.clone())),
                        _ => {
                            return Err(parse_error(
                                line,
                                format!("`{expr}` indexes a bit register more than once"),
                            ));
                        }
                    },
                    line,
                };
                Ok(Some(ClassicalExpr::Bit(self.bit_of(&operand)?)))
            }
            _ => Ok(None),
        }
    }

    /// Whether `expr` reads a runtime variable or a classical bit, which is
    /// what keeps it from folding at parse time.
    pub(super) fn reads_runtime(&self, expr: &Expr) -> bool {
        if self.dynamic.is_none() {
            return false;
        }
        reads_leaf(expr, &|leaf| self.runtime_leaf(leaf, 0)).unwrap_or(true)
    }

    /// Reject a runtime value where a parse-time constant belongs.
    pub(super) fn reject_runtime(&self, expr: &Expr, line: usize) -> Result<()> {
        if self.reads_runtime(expr) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`{expr}` reads a runtime value where a parse-time constant belongs"
                ),
                line,
            });
        }
        Ok(())
    }

    fn lower_expr(&self, expr: &Expr, line: usize) -> Result<ClassicalExpr> {
        lower(
            expr,
            line,
            &|leaf| self.runtime_leaf(leaf, line),
            &|folded| self.value_of(folded, line),
        )
    }

    fn lower_condition(&self, condition: &Condition, line: usize) -> Result<ClassicalExpr> {
        self.lower_expr(&condition_expr(condition)?, line)
    }

    fn decided(&self, condition: &Condition) -> Decided {
        let general = matches!(condition, Condition::Expr(_));
        let Ok(expr) = condition_expr(condition) else {
            return Decided::Bits;
        };
        let runtime = Cell::new(false);
        let bits = Cell::new(false);
        let _ = reads_leaf(&expr, &|leaf: &Expr| {
            match self.runtime_leaf(leaf, 0) {
                Ok(Some(ClassicalExpr::Var(_))) => runtime.set(true),
                Ok(Some(_)) | Err(_) => bits.set(true),
                Ok(None) => {}
            }
            Ok(None)
        });
        match (runtime.get(), bits.get()) {
            (true, _) => Decided::Runtime,
            (false, true) if general => Decided::Runtime,
            (false, true) => Decided::Bits,
            (false, false) => Decided::Static,
        }
    }

    // ---------------------------------------------------------- declarations

    fn runtime_type(&self, ty: &str, width: Option<&Expr>, line: usize) -> Result<RuntimeType> {
        let bits = match width {
            None => 64,
            Some(width) => {
                let bits = self.integer_of(width, line)?;
                u32::try_from(bits)
                    .ok()
                    .filter(|bits| (1..=64).contains(bits))
                    .ok_or_else(|| {
                        parse_error(line, format!("`{ty}[{bits}]` needs 1 to 64 bits"))
                    })?
            }
        };
        Ok(match ty {
            "int" => RuntimeType::Int { width: bits },
            "uint" => RuntimeType::Uint { width: bits },
            "bool" => RuntimeType::Bool,
            "float" => RuntimeType::Float,
            "angle" => RuntimeType::Angle,
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("a `{other}` whose value a measurement or a loop decides"),
                    line,
                });
            }
        })
    }

    /// Declare a variable that lives at runtime.
    pub(super) fn declare_runtime(
        &mut self,
        constant: bool,
        ty: &str,
        width: Option<&Expr<'a>>,
        name: &'a str,
        value: Option<&Expr<'a>>,
        line: usize,
    ) -> Result<()> {
        if constant {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("`const {name}` initialized from a runtime value"),
                line,
            });
        }
        let runtime_type = self.runtime_type(ty, width, line)?;
        self.reject_redeclaration(name, line)?;
        let value = match value {
            Some(value) => self.lower_expr(value, line)?,
            None => ClassicalExpr::Const(ClassicalValue::Int(0)),
        };
        let once = self.dynamic.as_ref().is_some_and(|state| state.repeat == 0);
        let var = match (&value, once) {
            (ClassicalExpr::Const(initial), true) => {
                self.state().builder.declare(name, runtime_type, *initial)
            }
            _ => {
                self.require_streaming("a runtime declaration", line)?;
                let builder = &mut self.state().builder;
                let var = builder.declare(name, runtime_type, ClassicalValue::Int(0));
                builder.assign(var, value);
                var
            }
        };
        let kind = match runtime_type {
            RuntimeType::Bool => ClassicalType::Bool,
            RuntimeType::Float | RuntimeType::Angle => ClassicalType::Float,
            _ => ClassicalType::Int,
        };
        self.classical.insert(
            name,
            ClassicalDecl {
                ty: kind,
                constant: false,
            },
        );
        self.state().vars.insert(name, var);
        Ok(())
    }

    pub(super) fn assign_runtime(
        &mut self,
        var: VarId,
        op: Option<AssignOp>,
        value: &Expr<'a>,
        line: usize,
    ) -> Result<()> {
        self.require_streaming("a runtime assignment", line)?;
        let mut value = self.lower_expr(value, line)?;
        if let Some(op) = op {
            value = ClassicalExpr::binary(compound_op(op), ClassicalExpr::Var(var), value);
        }
        self.state().builder.assign(var, value);
        Ok(())
    }

    // ------------------------------------------------------------- rotations

    /// Whether a gate argument reads a runtime value.
    pub(super) fn argument_is_runtime(&self, argument: &ast::Argument) -> bool {
        match argument {
            ast::Argument::Value(expr) => self.reads_runtime(expr),
            ast::Argument::Operand(operand) => {
                operand_expr(operand).is_ok_and(|expr| self.reads_runtime(&expr))
            }
        }
    }

    /// A gate whose one angle is computed at runtime.
    pub(super) fn exec_runtime_rotation(
        &mut self,
        modifiers: &[ast::Modifier],
        name: &str,
        params: &[ast::Argument],
        operands: &[Operand],
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let kind = match name {
            "rx" => Some(RotationKind::Rx),
            "ry" => Some(RotationKind::Ry),
            "rz" => Some(RotationKind::Rz),
            "p" | "phase" => Some(RotationKind::Phase),
            "rzz" => Some(RotationKind::Rzz),
            _ => None,
        };
        if self.def_defs.contains_key(name) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("a runtime value as an argument to def `{name}`"),
                line,
            });
        }
        let kind = match kind {
            Some(kind) if modifiers.is_empty() && !self.gate_defs.contains_key(name) => kind,
            _ => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "a runtime angle on `{name}`; one drives an unmodified `rx`, `ry`, \
                         `rz`, `p` or `rzz`"
                    ),
                    line,
                });
            }
        };
        let [param] = params else {
            return Err(parse_error(
                line,
                format!("`{name}` takes one angle, got {}", params.len()),
            ));
        };
        let angle = match param {
            ast::Argument::Value(expr) => self.lower_expr(expr, line)?,
            ast::Argument::Operand(operand) => self.lower_expr(&operand_expr(operand)?, line)?,
        };
        self.require_streaming("a runtime angle", line)?;
        let resolved: SmallVec<[SmallVec<[usize; 4]>; 4]> = operands
            .iter()
            .map(|operand| self.qubits_of(operand))
            .collect::<Result<_>>()?;
        let width = self.broadcast_length(&resolved, name, line)?;
        self.flush(std::mem::take(out));
        for at in 0..width {
            let qubits: SmallVec<[usize; 4]> = resolved
                .iter()
                .map(|entry| {
                    if entry.len() == 1 {
                        entry[0]
                    } else {
                        entry[at]
                    }
                })
                .collect();
            let arity = if kind == RotationKind::Rzz { 2 } else { 1 };
            if qubits.len() != arity {
                return Err(PrismError::GateArity {
                    gate: name.to_string(),
                    expected: arity,
                    got: qubits.len(),
                });
            }
            self.state()
                .builder
                .add_rotation(kind, &qubits, angle.clone());
        }
        Ok(())
    }

    // ---------------------------------------------------------- control flow

    /// Run a block whose execution the program decides at runtime, straight
    /// into the graph, in a scope of its own.
    fn branch_body(&mut self, block: &Block<'a>) -> Result<()> {
        let scope = self.open_scope();
        let was_nested = std::mem::replace(&mut self.nested, true);
        self.state().repeat += 1;
        let result = self.execute(block);
        self.state().repeat -= 1;
        self.nested = was_nested;
        self.close_scope(&scope);
        let leftover = result?;
        self.flush(leftover);
        Ok(())
    }

    pub(super) fn exec_while(
        &mut self,
        condition: &Condition<'a>,
        body: &Block<'a>,
        line: usize,
    ) -> Result<()> {
        self.require_streaming("`while`", line)?;
        let condition = self.lower_condition(condition, line)?;
        self.state()
            .builder
            .begin_while(format!("while at line {line}"), condition);
        self.state().loops.push(true);
        let result = self.branch_body(body);
        self.state().loops.pop();
        result?;
        self.state().builder.end()
    }

    pub(super) fn exec_loop_exit(&mut self, exit: bool, line: usize) -> Result<()> {
        let word = if exit { "break" } else { "continue" };
        let innermost = self
            .dynamic
            .as_ref()
            .and_then(|state| state.loops.last().copied());
        match innermost {
            Some(true) => {}
            Some(false) => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "`{word}` in a `for` loop, which unrolls at parse time; use `while`"
                    ),
                    line,
                });
            }
            None => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`{word}` outside a loop"),
                    line,
                });
            }
        }
        self.require_streaming(&format!("`{word}`"), line)?;
        let builder = &mut self.state().builder;
        if exit {
            builder.break_loop()
        } else {
            builder.continue_loop()
        }
    }

    /// An `if` under dynamic lowering: folded when it reads nothing measured, a
    /// guarded region when the instruction list can say its condition and its
    /// bodies open no block, and a branch in the graph otherwise.
    pub(super) fn exec_if_dynamic(
        &mut self,
        stmt: &Stmt<'a>,
        conditional: &ast::Conditional<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let decided = self.decided(&conditional.condition);
        let needs_graph = self
            .dynamic
            .as_ref()
            .is_some_and(|state| state.analysis.branches.contains(&key(stmt)));
        if decided == Decided::Static {
            let taken = self.value_of(&condition_expr(&conditional.condition)?, line)? != 0.0;
            let body = if taken {
                Some(&conditional.then_body)
            } else {
                conditional.else_body.as_ref()
            };
            if let Some(body) = body {
                out.extend(self.exec_box(body)?);
            }
            return Ok(());
        }
        let guard = if decided == Decided::Bits && !needs_graph {
            self.condition_of(&conditional.condition, line).ok()
        } else {
            None
        };
        let Some(guard) = guard else {
            return self.branch_if(conditional, line);
        };
        let then_instrs = self.region(&conditional.then_body)?;
        let Some(else_body) = &conditional.else_body else {
            out.extend(guarded(guard, then_instrs));
            return Ok(());
        };
        if !crate::circuit::body_writes_condition_bits(&then_instrs, &guard) {
            let else_instrs = self.region(else_body)?;
            out.extend(guarded(guard.clone(), then_instrs));
            out.extend(guarded(guard.negate(), else_instrs));
            return Ok(());
        }
        // The guard pair would re-read bits the `then` body overwrites, and a
        // branch reads them once.
        self.require_streaming("an `else` whose `if` body measures its own condition", line)?;
        let condition = self.lower_condition(&conditional.condition, line)?;
        self.flush(std::mem::take(out));
        self.state().builder.begin_if(condition);
        self.flush(then_instrs);
        self.state().builder.begin_else()?;
        self.branch_body(else_body)?;
        self.state().builder.end()
    }

    fn branch_if(&mut self, conditional: &ast::Conditional<'a>, line: usize) -> Result<()> {
        self.require_streaming("an `if` over a runtime value", line)?;
        let condition = self.lower_condition(&conditional.condition, line)?;
        self.state().builder.begin_if(condition);
        self.branch_body(&conditional.then_body)?;
        if let Some(else_body) = &conditional.else_body {
            self.state().builder.begin_else()?;
            self.branch_body(else_body)?;
        }
        self.state().builder.end()
    }

    /// Whether a `switch` lowers to branches in the graph rather than guards.
    pub(super) fn switch_needs_graph(&self, stmt: &Stmt, operand: &Operand) -> bool {
        let Some(state) = self.dynamic.as_ref() else {
            return false;
        };
        state.analysis.branches.contains(&key(stmt))
            || operand
                .register()
                .is_some_and(|name| self.runtime_var(name).is_some())
    }

    /// A `switch` as a chain of branches, one per arm, the `default` in the
    /// last `else`.
    pub(super) fn exec_switch_branches(
        &mut self,
        operand: &Operand<'a>,
        arms: &[ast::SwitchArm<'a>],
        line: usize,
    ) -> Result<()> {
        self.require_streaming("a `switch` over a runtime value", line)?;
        let value = self.lower_expr(&operand_expr(operand)?, line)?;
        let mut seen: Vec<i64> = Vec::new();
        let mut default = None;
        let mut cases = Vec::new();
        for arm in arms {
            let Some(labels) = &arm.labels else {
                if default.replace(&arm.body).is_some() {
                    return Err(parse_error(
                        arm.line,
                        "`switch` has more than one `default` arm",
                    ));
                }
                continue;
            };
            let mut test: Option<ClassicalExpr> = None;
            for label in labels {
                let label = self.integer_of(label, arm.line)?;
                if seen.contains(&label) {
                    return Err(parse_error(
                        arm.line,
                        format!("`switch` case label {label} appears twice"),
                    ));
                }
                seen.push(label);
                let equal = ClassicalExpr::binary(
                    dynamic::BinaryOp::Eq,
                    value.clone(),
                    ClassicalExpr::from(label),
                );
                test = Some(match test {
                    None => equal,
                    Some(test) => ClassicalExpr::binary(dynamic::BinaryOp::Or, test, equal),
                });
            }
            cases.push((test.expect("a case names a label"), &arm.body));
        }
        for (test, body) in &cases {
            self.state().builder.begin_if(test.clone());
            self.branch_body(body)?;
            self.state().builder.begin_else()?;
        }
        if let Some(body) = default {
            self.branch_body(body)?;
        }
        for _ in &cases {
            self.state().builder.end()?;
        }
        Ok(())
    }
}

/// Which declarations live at runtime and which branches need the graph,
/// keyed by statement.
#[derive(Default)]
pub(super) struct RuntimeAnalysis {
    runtime: HashSet<usize>,
    branches: HashSet<usize>,
}

#[derive(Clone, Copy)]
enum Binding {
    Var { decl: usize, depth: usize },
    Constant,
    Array { decl: usize, depth: usize },
    Bits,
    Other,
}

struct Walk<'s, 'a> {
    runtime: &'s mut HashSet<usize>,
    branches: HashSet<usize>,
    changed: bool,
    scopes: Vec<HashMap<&'a str, Binding>>,
}

#[derive(Clone, Copy, Default)]
struct Reads {
    runtime: bool,
    bits: bool,
}

impl Reads {
    fn any(self) -> bool {
        self.runtime || self.bits
    }

    fn or(self, other: Reads) -> Reads {
        Reads {
            runtime: self.runtime || other.runtime,
            bits: self.bits || other.bits,
        }
    }
}

impl RuntimeAnalysis {
    /// Iterate to a fixed point: a variable assigned from a runtime one is
    /// itself runtime, which can make a third one so on a later pass.
    pub(super) fn of(program: &[Stmt]) -> RuntimeAnalysis {
        let mut runtime = HashSet::new();
        loop {
            let mut walk = Walk {
                runtime: &mut runtime,
                branches: HashSet::new(),
                changed: false,
                scopes: vec![HashMap::new()],
            };
            walk.block(program, 0);
            if !walk.changed {
                let branches = walk.branches;
                return RuntimeAnalysis { runtime, branches };
            }
        }
    }
}

impl<'a> Walk<'_, 'a> {
    fn bind(&mut self, name: &'a str, binding: Binding) {
        self.scopes
            .last_mut()
            .expect("a scope is always open")
            .insert(name, binding);
    }

    fn lookup(&self, name: &str) -> Binding {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
            .unwrap_or(Binding::Other)
    }

    fn mark(&mut self, decl: usize) {
        if self.runtime.insert(decl) {
            self.changed = true;
        }
    }

    fn name_reads(&self, name: &str) -> Reads {
        match self.lookup(name) {
            Binding::Var { decl, .. } | Binding::Array { decl, .. } => Reads {
                runtime: self.runtime.contains(&decl),
                bits: false,
            },
            Binding::Bits => Reads {
                runtime: false,
                bits: true,
            },
            Binding::Constant | Binding::Other => Reads::default(),
        }
    }

    fn expr_reads(&self, expr: &Expr) -> Reads {
        match expr {
            Expr::Number(_) | Expr::Duration(_) => Reads::default(),
            Expr::Ident(name) => self.name_reads(name),
            Expr::Element(element) => element
                .indices
                .iter()
                .fold(self.name_reads(element.array), |reads, index| {
                    reads.or(self.expr_reads(index))
                }),
            Expr::Negate(inner) | Expr::Not(inner) | Expr::BitNot(inner) => self.expr_reads(inner),
            Expr::Binary { left, right, .. } => self.expr_reads(left).or(self.expr_reads(right)),
            Expr::Call(call) => call.args.iter().fold(Reads::default(), |reads, arg| {
                reads.or(self.expr_reads(arg))
            }),
        }
    }

    fn argument_reads(&self, argument: &ast::Argument) -> Reads {
        match argument {
            ast::Argument::Value(expr) => self.expr_reads(expr),
            ast::Argument::Operand(operand) => match operand_expr(operand) {
                Ok(expr) => self.expr_reads(&expr),
                Err(_) => Reads::default(),
            },
        }
    }

    fn decided(&self, condition: &Condition) -> Decided {
        let reads = match condition_expr(condition) {
            Ok(expr) => self.expr_reads(&expr),
            Err(_) => Reads {
                runtime: false,
                bits: true,
            },
        };
        match (reads.runtime, reads.bits) {
            (true, _) => Decided::Runtime,
            (false, true) if matches!(condition, Condition::Expr(_)) => Decided::Runtime,
            (false, true) => Decided::Bits,
            (false, false) => Decided::Static,
        }
    }

    /// A write to the variable or array `binding`, at runtime depth `depth`.
    fn write(&mut self, binding: Binding, depth: usize, reads: Reads) -> bool {
        match binding {
            Binding::Var { decl, depth: home } | Binding::Array { decl, depth: home } => {
                if depth > home || reads.any() {
                    self.mark(decl);
                }
                self.runtime.contains(&decl)
            }
            _ => false,
        }
    }

    fn scoped(&mut self, block: &[Stmt<'a>], depth: usize) -> bool {
        self.scopes.push(HashMap::new());
        let needs = self.block(block, depth);
        self.scopes.pop();
        needs
    }

    /// Walk `block` at runtime depth `depth`, reporting whether anything in it
    /// opens a block of the graph.
    fn block(&mut self, block: &[Stmt<'a>], depth: usize) -> bool {
        let mut needs = false;
        for stmt in block {
            needs |= self.stmt(stmt, depth);
        }
        needs
    }

    fn stmt(&mut self, stmt: &Stmt<'a>, depth: usize) -> bool {
        match &stmt.kind {
            StmtKind::RegisterDecl { kind, name, .. } => {
                let binding = match kind {
                    ast::RegisterKind::Classical => Binding::Bits,
                    ast::RegisterKind::Qubit => Binding::Other,
                };
                self.bind(name, binding);
                false
            }
            StmtKind::OutputDecl { name, .. } => {
                self.bind(name, Binding::Bits);
                false
            }
            StmtKind::InputDecl { name, .. } => {
                self.bind(name, Binding::Other);
                false
            }
            StmtKind::ClassicalDecl {
                constant,
                name,
                value,
                ..
            } => {
                if *constant {
                    self.bind(name, Binding::Constant);
                    return false;
                }
                let decl = key(stmt);
                if value
                    .as_ref()
                    .is_some_and(|value| self.expr_reads(value).any())
                {
                    self.mark(decl);
                }
                self.bind(name, Binding::Var { decl, depth });
                self.runtime.contains(&decl)
            }
            StmtKind::Assign { target, value, .. } => {
                let reads = self.expr_reads(value);
                self.write(self.lookup(target), depth, reads)
            }
            StmtKind::CallAssign(assign) if assign.target.index.is_none() => {
                let Some(target) = assign.target.register() else {
                    return false;
                };
                let reads = assign.args.iter().fold(Reads::default(), |reads, arg| {
                    reads.or(self.argument_reads(arg))
                });
                self.write(self.lookup(target), depth, reads)
            }
            StmtKind::ArrayDecl(decl) => {
                self.bind(
                    decl.name,
                    Binding::Array {
                        decl: key(stmt),
                        depth,
                    },
                );
                false
            }
            StmtKind::ElementAssign(assign) => {
                let reads = assign
                    .indices
                    .iter()
                    .fold(self.expr_reads(&assign.value), |reads, index| {
                        reads.or(self.expr_reads(index))
                    });
                self.write(self.lookup(assign.array), depth, reads);
                false
            }
            StmtKind::Alias { name, sources } => {
                let bits = sources.iter().any(|source| {
                    source
                        .register()
                        .is_some_and(|name| matches!(self.lookup(name), Binding::Bits))
                });
                self.bind(name, if bits { Binding::Bits } else { Binding::Other });
                false
            }
            StmtKind::Call { params, .. } => {
                params.iter().any(|param| self.argument_reads(param).any())
            }
            StmtKind::If(conditional) => {
                let decided = self.decided(&conditional.condition);
                let inner = depth + usize::from(decided != Decided::Static);
                let mut needs = self.scoped(&conditional.then_body, inner);
                if let Some(body) = &conditional.else_body {
                    needs |= self.scoped(body, inner);
                }
                needs |= decided == Decided::Runtime;
                if needs {
                    self.branches.insert(key(stmt));
                }
                needs
            }
            StmtKind::Switch { operand, arms } => {
                let reads = operand
                    .register()
                    .map_or(Reads::default(), |name| self.name_reads(name));
                let mut needs = reads.runtime;
                for arm in arms {
                    needs |= self.scoped(&arm.body, depth + 1);
                }
                if needs {
                    self.branches.insert(key(stmt));
                }
                needs
            }
            StmtKind::While { body, .. } => {
                self.scoped(body, depth + 1);
                true
            }
            StmtKind::Break | StmtKind::Continue => true,
            StmtKind::For { variable, body, .. } => {
                self.scopes.push(HashMap::new());
                self.bind(variable, Binding::Constant);
                let needs = self.block(body, depth);
                self.scopes.pop();
                needs
            }
            StmtKind::Box { body, .. } => self.scoped(body, depth),
            _ => false,
        }
    }
}
