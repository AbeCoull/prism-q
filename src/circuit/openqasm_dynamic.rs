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
use crate::circuit::qasm::stream::{Stream, with_stack};
use crate::circuit::qasm::{lexer, parser as syntax};

/// Budget units a block of an unrolled loop costs, sized so a pass of blocks
/// holds about the memory per unit of an ordinary unrolled pass.
const BLOCK_UNITS: usize = 2;

#[cfg(test)]
thread_local! {
    /// Statements the runtime analysis has walked on this thread.
    pub(super) static WALKED: Cell<usize> = const { Cell::new(0) };
}

/// What the evaluator carries while lowering into a control-flow graph.
pub(super) struct DynamicState<'a> {
    builder: DynamicProgramBuilder,
    analysis: RuntimeAnalysis,
    /// Runtime variables in scope, by name.
    pub(super) vars: HashMap<&'a str, VarId>,
    pub(super) bit_values: HashMap<&'a str, usize>,
    /// Nesting of guarded regions being collected into one instruction, where
    /// nothing may open a block.
    collecting: usize,
    /// Constructs a declaration may run under other than once per shot: loop
    /// bodies and branches decided at runtime.
    repeat: usize,
    /// Blocks the passes of unrolled loops have charged to the budget.
    charged_blocks: usize,
}

/// Statement keys: the address of the statement in the tree, which every pass
/// over one body sees unchanged.
fn key(stmt: &Stmt) -> usize {
    stmt as *const Stmt as usize
}

fn scalar_runtime_type(kind: ClassicalType, line: usize) -> Result<RuntimeType> {
    Ok(match kind {
        ClassicalType::Int(ty) => ty,
        ClassicalType::Float(width) => RuntimeType::Float { width },
        ClassicalType::Bool => RuntimeType::Bool,
        ClassicalType::Angle(None) => RuntimeType::Angle,
        ClassicalType::Angle(Some(_)) | ClassicalType::Duration | ClassicalType::Stretch => {
            return Err(PrismError::UnsupportedConstruct {
                construct: "a runtime duration or fixed-width angle in a subroutine".to_string(),
                line,
            });
        }
    })
}

/// Parse an OpenQASM 3.0 program that may need runtime control flow into a
/// [`DynamicProgram`].
///
/// Reads everything [`parse`] reads, plus `while` loops, `break`, `continue` and
/// `end` under a measured branch, conditions and assignments over the full classical operator set
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
/// builtin function over a runtime value, and a measurement into a variable
/// rather than a bit.
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
    with_stack(|| parse_dynamic_within(input, MAX_EXPANSION))
}

/// [`parse_dynamic`] with `budget` units of expansion in place of the default.
pub(super) fn parse_dynamic_within(input: &str, budget: usize) -> Result<DynamicProgram> {
    let tokens = lexer::tokenize(input)?;
    let program = syntax::parse_dynamic_program(&tokens)?;
    let mut parser = Parser::new_with(input, Dialect::Native);
    parser.expansion_left.set(budget);
    parser.dynamic = Some(Box::new(DynamicState {
        builder: DynamicProgramBuilder::new(0, 0),
        analysis: RuntimeAnalysis::of(&program),
        vars: HashMap::new(),
        bit_values: HashMap::new(),
        collecting: 0,
        repeat: 0,
        charged_blocks: 0,
    }));
    if let Some(width) = super::eval::physical_width(&program)? {
        parser.physical = true;
        parser.total_qubits = width;
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
                    match syntax_expr::eval(index, 1, None)? {
                        ClassicalValue::Int(at) if at >= 0 => Some(ClassicalExpr::Bit(at as usize)),
                        at => {
                            let at = at.to_float();
                            return Err(parse_error(1, format!("`c[{at}]` is not a bit")));
                        }
                    }
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

pub(super) fn bits_value(bits: &[usize], line: usize) -> Result<ClassicalExpr> {
    if bits.is_empty() || bits.len() > 64 {
        return Err(PrismError::UnsupportedConstruct {
            construct: "a subroutine bit value wider than 64 bits".to_string(),
            line,
        });
    }
    if bits.windows(2).all(|pair| pair[1] == pair[0] + 1) {
        return register_read(bits[0], bits.len(), line);
    }
    let mut value = ClassicalExpr::Bit(bits[0]);
    for (at, &bit) in bits.iter().enumerate().skip(1) {
        value = ClassicalExpr::binary(
            dynamic::BinaryOp::BitOr,
            value,
            ClassicalExpr::binary(
                dynamic::BinaryOp::Shl,
                ClassicalExpr::Bit(bit),
                ClassicalExpr::Const(ClassicalValue::Int(at as i128)),
            ),
        );
    }
    Ok(value)
}

/// Whether some leaf of `expr` is a runtime value according to `leaf`.
fn reads_leaf(expr: &Expr, leaf: &impl Fn(&Expr) -> Result<Option<ClassicalExpr>>) -> Result<bool> {
    with_stack(|| {
        Ok(match expr {
            Expr::Number(_) | Expr::Duration(_) => false,
            Expr::Ident(_) | Expr::Runtime(_) => leaf(expr)?.is_some(),
            Expr::Element(element) => {
                leaf(expr)?.is_some()
                    || element.indices.iter().try_fold(false, |seen, index| {
                        Ok::<_, PrismError>(seen || reads_leaf(index, leaf)?)
                    })?
            }
            Expr::Negate(inner) | Expr::Not(inner) | Expr::BitNot(inner) => {
                reads_leaf(inner, leaf)?
            }
            Expr::Binary { left, right, .. } => reads_leaf(left, leaf)? || reads_leaf(right, leaf)?,
            Expr::Call(call) => call.args.iter().try_fold(false, |seen, arg| {
                Ok::<_, PrismError>(seen || reads_leaf(arg, leaf)?)
            })?,
            Expr::Cast(cast) => reads_leaf(&cast.value, leaf)?,
        })
    })
}

/// Lower `expr`, folding every subtree with no runtime leaf through `fold`.
fn lower(
    expr: &Expr,
    line: usize,
    leaf: &impl Fn(&Expr) -> Result<Option<ClassicalExpr>>,
    fold: &impl Fn(&Expr) -> Result<ClassicalValue>,
) -> Result<ClassicalExpr> {
    with_stack(|| {
        if !reads_leaf(expr, leaf)? {
            return Ok(ClassicalExpr::Const(fold(expr)?));
        }
        let recurse = |inner: &Expr| lower(inner, line, leaf, fold);
        Ok(match expr {
            Expr::Ident(_) | Expr::Element(_) | Expr::Runtime(_) => match leaf(expr)? {
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
                ClassicalExpr::binary(op.runtime(), recurse(left)?, recurse(right)?)
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
            Expr::Cast(cast) => {
                let width = match &cast.width {
                    Some(width) => Some(cast.ty.width(fold(width)?, line)?),
                    None => None,
                };
                ClassicalExpr::cast(cast.ty.runtime(width), recurse(&cast.value)?)
            }
            Expr::Number(_) | Expr::Duration(_) => unreachable!("a literal reads no runtime leaf"),
        })
    })
}

/// The expression a static condition form spells, for a condition over
/// variables rather than bits.
pub(super) fn condition_expr<'a>(condition: &Condition<'a>) -> Result<Expr<'a>> {
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
                None => binary(BinaryOp::Eq, parity, Expr::Number(ClassicalValue::Int(1))),
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
    pub(super) fn resolve_short_circuit(
        &mut self,
        op: BinaryOp,
        left: &Expr<'a>,
        right: &Expr<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<Expr<'a>> {
        self.require_streaming("a subroutine call under short-circuit evaluation", line)?;
        self.flush(std::mem::take(out));
        let left = self.lower_expr(left, line)?;
        let var = self.state().builder.declare(
            "short-circuit result",
            RuntimeType::Bool,
            ClassicalValue::Bool(false),
        );
        self.state()
            .builder
            .assign(var, ClassicalExpr::cast(RuntimeType::Bool, left.clone()));
        let condition = if op == BinaryOp::And {
            left
        } else {
            ClassicalExpr::unary(UnaryOp::Not, left)
        };
        self.state().builder.begin_if(condition);
        let right = self.resolve_calls(right, line, out)?;
        self.flush(std::mem::take(out));
        let value = self.lower_expr(&right, line)?;
        self.state()
            .builder
            .assign(var, ClassicalExpr::cast(RuntimeType::Bool, value));
        self.state().builder.end()?;
        Ok(Expr::Runtime(var))
    }
    pub(super) fn exec_runtime_return(
        &mut self,
        value: Option<&ast::ReturnValue<'a>>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let frame = self.def_result.as_ref().expect("runtime call frame");
        let var = frame.result_var;
        let bits = frame.bits.clone();
        match (value, var) {
            (Some(ast::ReturnValue::Value(value)), Some(var)) => {
                let value = self.resolve_calls(value, line, out)?;
                let value = self.lower_expr(&value, line)?;
                self.flush(std::mem::take(out));
                self.state().builder.assign(var, value);
            }
            (Some(ast::ReturnValue::Measure(source)), None) => {
                out.extend(Self::build_measurements(
                    self.qubits_of(source)?,
                    bits,
                    line,
                )?);
            }
            (Some(ast::ReturnValue::Value(value)), None) if !bits.is_empty() => {
                let value = self.resolve_calls(value, line, out)?;
                let value = self.lower_expr(&value, line)?;
                if matches!(value, ClassicalExpr::Const(ClassicalValue::Float(_))) {
                    return Err(parse_error(line, "a bit result needs an integer or bool"));
                }
                self.flush(std::mem::take(out));
                self.write_classical_bits(&bits, value, line)?;
            }
            (Some(ast::ReturnValue::Measure(_)), Some(_)) => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "a measurement returned as a classical scalar".to_string(),
                    line,
                });
            }
            _ => {}
        }
        self.flush(std::mem::take(out));
        self.state().builder.return_from_call()?;
        self.def_result
            .as_mut()
            .expect("runtime call frame")
            .returned = true;
        Ok(())
    }

    pub(super) fn assign_bit_parameter(
        &mut self,
        assign: &ast::ElementAssign<'a>,
        line: usize,
    ) -> Result<()> {
        self.require_streaming("a write to a copied bit parameter", line)?;
        let width = self.state().bit_values[assign.array];
        let [index] = assign.indices.as_slice() else {
            return Err(parse_error(line, "a bit parameter takes one index"));
        };
        let operand = Operand {
            name: ast::OperandName::Register(assign.array),
            index: Some(ast::Index::Single(index.clone())),
            line,
        };
        let picks = self.pick_indices(
            width,
            operand.index.as_ref().expect("bit index"),
            ast::RegisterKind::Classical,
            &operand,
        )?;
        let index = picks[0];
        let var = self
            .runtime_var(assign.array)
            .expect("copied bit parameter");
        let word = ClassicalExpr::Var(var);
        let bit = ClassicalExpr::binary(
            dynamic::BinaryOp::BitAnd,
            ClassicalExpr::binary(
                dynamic::BinaryOp::Shr,
                word.clone(),
                ClassicalExpr::Const(ClassicalValue::Int(index as i128)),
            ),
            ClassicalExpr::Const(ClassicalValue::Int(1)),
        );
        let mut value = self.lower_expr(&assign.value, line)?;
        if let Some(op) = assign.op {
            value = ClassicalExpr::binary(op.binary().runtime(), bit, value);
        }
        let value = ClassicalExpr::binary(
            dynamic::BinaryOp::BitAnd,
            value,
            ClassicalExpr::Const(ClassicalValue::Int(1)),
        );
        let value = ClassicalExpr::binary(
            dynamic::BinaryOp::Shl,
            value,
            ClassicalExpr::Const(ClassicalValue::Int(index as i128)),
        );
        let cleared = ClassicalExpr::binary(
            dynamic::BinaryOp::BitAnd,
            word,
            ClassicalExpr::Const(ClassicalValue::Int(!(1i128 << index))),
        );
        self.state().builder.assign(
            var,
            ClassicalExpr::binary(dynamic::BinaryOp::BitOr, cleared, value),
        );
        Ok(())
    }
    pub(super) fn def_is_runtime(
        &self,
        name: &str,
        params: &[ast::DefParam],
        args: &[ast::Argument],
    ) -> bool {
        self.dynamic.as_ref().is_some_and(|state| {
            state.analysis.runtime_defs.contains(name)
                || params.iter().zip(args).any(|(param, arg)| {
                    matches!(param, ast::DefParam::Value { .. }) && self.argument_is_runtime(arg)
                })
                || args.iter().any(|arg| {
                    match arg {
                        ast::Argument::Operand(operand) => operand.register(),
                        ast::Argument::Value(Expr::Ident(name)) => Some(*name),
                        ast::Argument::Value(Expr::Element(element)) => Some(element.array),
                        _ => None,
                    }
                    .is_some_and(|name| state.bit_values.contains_key(name))
                })
        })
    }

    pub(super) fn runtime_bit_argument(
        &self,
        argument: &ast::Argument,
        line: usize,
    ) -> Result<Option<(usize, ClassicalExpr)>> {
        let operand = match argument {
            ast::Argument::Operand(operand) => std::borrow::Cow::Borrowed(operand),
            ast::Argument::Value(Expr::Ident(name)) => std::borrow::Cow::Owned(Operand {
                name: ast::OperandName::Register(name),
                index: None,
                line,
            }),
            ast::Argument::Value(Expr::Element(element)) if element.indices.len() == 1 => {
                std::borrow::Cow::Owned(Operand {
                    name: ast::OperandName::Register(element.array),
                    index: Some(ast::Index::Single(element.indices[0].clone())),
                    line,
                })
            }
            _ => return Ok(None),
        };
        let Some((name, width)) = operand.register().and_then(|name| {
            self.dynamic
                .as_ref()
                .and_then(|state| state.bit_values.get(name).map(|width| (name, *width)))
        }) else {
            return Ok(None);
        };
        let value = ClassicalExpr::Var(self.runtime_var(name).expect("bit parameter binding"));
        let Some(index) = &operand.index else {
            return Ok(Some((width, value)));
        };
        let picks = self.pick_indices(width, index, ast::RegisterKind::Classical, &operand)?;
        if picks.is_empty() || picks.len() > 64 {
            return Err(PrismError::UnsupportedConstruct {
                construct: "a subroutine bit value wider than 64 bits".to_string(),
                line,
            });
        }
        let bit = |index: usize| {
            ClassicalExpr::binary(
                dynamic::BinaryOp::BitAnd,
                ClassicalExpr::binary(
                    dynamic::BinaryOp::Shr,
                    value.clone(),
                    ClassicalExpr::Const(ClassicalValue::Int(index as i128)),
                ),
                ClassicalExpr::Const(ClassicalValue::Int(1)),
            )
        };
        let mut selected = bit(picks[0]);
        for (position, &index) in picks.iter().enumerate().skip(1) {
            selected = ClassicalExpr::binary(
                dynamic::BinaryOp::BitOr,
                selected,
                ClassicalExpr::binary(
                    dynamic::BinaryOp::Shl,
                    bit(index),
                    ClassicalExpr::Const(ClassicalValue::Int(position as i128)),
                ),
            );
        }
        Ok(Some((picks.len(), selected)))
    }

    pub(super) fn execute_runtime_def(
        &mut self,
        name: &str,
        def: &DefDefinition<'a>,
        mut sub: Parser<'a>,
        args: Vec<(&'a str, ClassicalType, ClassicalExpr)>,
        line: usize,
    ) -> Result<Option<Expr<'a>>> {
        self.require_streaming("a runtime subroutine call", line)?;
        self.charge(args.len() + 1, line)?;
        let runtime_params = args
            .iter()
            .filter_map(|(name, _, value)| {
                (!matches!(value, ClassicalExpr::Const(_))).then_some(*name)
            })
            .collect::<Vec<_>>();
        let state = self.dynamic.as_mut().expect("runtime call");
        let analysis = RuntimeAnalysis::for_def(
            &def.body,
            &def.args,
            def.result.as_ref().is_some_and(|result| result.ty != "bit"),
            Rc::clone(&state.analysis.runtime_defs),
            &runtime_params,
        );
        let mut state = self.dynamic.take().expect("runtime call");
        let caller_analysis = std::mem::replace(&mut state.analysis, analysis);
        let caller_vars = std::mem::take(&mut state.vars);
        let caller_bits = std::mem::take(&mut state.bit_values);
        let caller_repeat = state.repeat;
        state.repeat += 1;
        sub.dynamic = Some(state);
        let mut ended = false;
        let result = (|| {
            let result_var =
                if let Some(result) = def.result.as_ref().filter(|result| result.ty != "bit") {
                    let kind = sub.classical_type(result.ty, result.width.as_ref(), line)?;
                    let ty = scalar_runtime_type(kind, line)?;
                    Some(sub.state().builder.declare(
                        format!("result of {name}"),
                        ty,
                        ClassicalValue::Int(0),
                    ))
                } else {
                    None
                };
            let frame = sub.def_result.as_mut().expect("call frame");
            frame.runtime = true;
            frame.result_var = result_var;
            sub.state().builder.begin_call();
            let mutable = def
                .args
                .iter()
                .filter_map(|param| match param {
                    ast::DefParam::Value { name, .. }
                        if sub
                            .state()
                            .analysis
                            .runtime
                            .contains(&(param as *const ast::DefParam as usize)) =>
                    {
                        Some(*name)
                    }
                    _ => None,
                })
                .collect::<HashSet<_>>();
            let bit_params = def
                .args
                .iter()
                .filter_map(|param| match param {
                    ast::DefParam::Bit { name, .. } => Some(*name),
                    _ => None,
                })
                .collect::<HashSet<_>>();
            for (name, kind, value) in args {
                Rc::make_mut(&mut sub.cregs).remove(name);
                Rc::make_mut(&mut sub.aliases).remove(name);
                sub.params.remove(name);
                Rc::make_mut(&mut sub.classical).insert(
                    name,
                    ClassicalDecl {
                        ty: kind,
                        constant: false,
                        value: None,
                    },
                );
                match value {
                    ClassicalExpr::Const(value) if !mutable.contains(name) => {
                        sub.param_vars
                            .as_mut()
                            .expect("callee bindings")
                            .insert(name, kind.store(value));
                    }
                    value => {
                        let ty = scalar_runtime_type(kind, line)?;
                        let var = sub
                            .state()
                            .builder
                            .declare(name, ty, ClassicalValue::Int(0));
                        sub.state().builder.assign(var, value);
                        sub.state().vars.insert(name, var);
                        if bit_params.contains(name) {
                            let RuntimeType::Uint { width } = ty else {
                                unreachable!("bit argument type")
                            };
                            sub.state().bit_values.insert(name, width as usize);
                        }
                        sub.param_vars
                            .as_mut()
                            .expect("callee bindings")
                            .remove(name);
                    }
                }
            }
            let instructions = sub.execute(&def.body)?;
            sub.flush(instructions);
            ended = sub.exit == Some(Exit::End);
            let fallthrough = sub.state().builder.end_call()?;
            if def.result.is_some() && fallthrough {
                return Err(parse_error(
                    line,
                    format!("def `{name}` reaches its end without a result"),
                ));
            }
            Ok(result_var.map(Expr::Runtime))
        })();
        let mut state = sub.dynamic.take().expect("callee state");
        state.analysis = caller_analysis;
        state.vars = caller_vars;
        state.bit_values = caller_bits;
        state.repeat = caller_repeat;
        self.dynamic = Some(state);
        if ended {
            self.exit = Some(Exit::End);
        }
        result
    }
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

    /// `target = value` or `target op= value` where `target` names classical
    /// bits, which only a dynamic program can write.
    pub(super) fn assign_bits(
        &mut self,
        target: &Operand<'a>,
        op: Option<AssignOp>,
        value: &Expr<'a>,
        line: usize,
    ) -> Result<()> {
        if self.dynamic.is_none() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "a write to classical bits `{}`; the instruction list has no classical \
                     write, and `openqasm::parse_dynamic` carries one",
                    target.describe()
                ),
                line,
            });
        }
        let bits = self.bits_of(target)?;
        let mut lowered = self.lower_expr(value, line)?;
        if let Some(op) = op {
            let current = self.lower_expr(&operand_expr(target)?, line)?;
            lowered = ClassicalExpr::binary(op.binary().runtime(), current, lowered);
        }
        match lowered {
            ClassicalExpr::Const(ClassicalValue::Float(_)) => Err(parse_error(
                line,
                format!(
                    "`{}` holds bits and takes an integer or a bool, not `{value}`",
                    target.describe()
                ),
            )),
            ClassicalExpr::Const(value) if bits.len() > 64 => {
                let word = value.to_int();
                let values: Vec<bool> = (0..bits.len())
                    .map(|at| {
                        if at < 128 {
                            (word >> at) & 1 == 1
                        } else {
                            word < 0
                        }
                    })
                    .collect();
                self.write_constant_bits(&bits, &values, line)
            }
            lowered => self.write_classical_bits(&bits, lowered, line),
        }
    }

    /// Write the low bits of `value` to `bits`, least significant first.
    pub(super) fn write_classical_bits(
        &mut self,
        bits: &[usize],
        value: ClassicalExpr,
        line: usize,
    ) -> Result<()> {
        self.require_streaming("a write to a classical bit", line)?;
        if bits.len() > 64 {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "a write of {} bits from a runtime value, past the 64 a value holds",
                    bits.len()
                ),
                line,
            });
        }
        self.charge(1, line)?;
        self.state().builder.write_bits(bits, value);
        Ok(())
    }

    /// Write `values` to `bits`, one write per 64 bits.
    pub(super) fn write_constant_bits(
        &mut self,
        bits: &[usize],
        values: &[bool],
        line: usize,
    ) -> Result<()> {
        for (bits, values) in bits.chunks(64).zip(values.chunks(64)) {
            let word = values
                .iter()
                .rev()
                .fold(0u64, |acc, &bit| (acc << 1) | u64::from(bit));
            let value = ClassicalExpr::Const(ClassicalValue::Int(word.into()));
            self.write_classical_bits(bits, value, line)?;
        }
        Ok(())
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
            state.repeat += 1;
        }
    }

    pub(super) fn leave_static_loop(&mut self) {
        if let Some(state) = self.dynamic.as_mut() {
            state.repeat -= 1;
        }
    }

    pub(super) fn in_while(&self) -> bool {
        self.loops.contains(&Loop::While)
    }

    /// Whether a `for` needs its passes as blocks of the graph, because a
    /// measurement decides whether one of them breaks or continues.
    pub(super) fn unrolls_into_graph(&self, stmt: &Stmt) -> bool {
        self.streaming()
            && self
                .dynamic
                .as_ref()
                .is_some_and(|state| state.analysis.unrolled.contains(&key(stmt)))
    }

    pub(super) fn graph_depth(&self) -> usize {
        self.dynamic
            .as_ref()
            .map_or(0, |state| state.builder.depth())
    }

    pub(super) fn begin_unrolled(&mut self) {
        self.state().builder.begin_unrolled();
    }

    pub(super) fn next_pass(&mut self) -> Result<()> {
        self.state().builder.next_pass()
    }

    pub(super) fn end_unrolled(&mut self) -> Result<()> {
        self.state().builder.end()
    }

    /// Whether a runtime `continue` has jumped from the pass being built to the
    /// next one, so a parse-time exit cannot stop the unrolling.
    pub(super) fn pass_continued(&self) -> bool {
        self.dynamic
            .as_ref()
            .is_some_and(|state| state.builder.pass_continued())
    }

    /// Whether a runtime `break` has jumped out of the innermost unrolled loop,
    /// so its exit is reached even where a parse-time `return` ended a pass.
    pub(super) fn loop_broke(&self) -> bool {
        self.dynamic
            .as_ref()
            .is_some_and(|state| state.builder.loop_broke())
    }

    pub(super) fn break_unrolled(&mut self) -> Result<()> {
        self.state().builder.break_loop()
    }

    /// The block count and the blocks charged so far, read at the start of a pass.
    pub(super) fn pass_mark(&self) -> (usize, usize) {
        self.dynamic.as_ref().map_or((0, 0), |state| {
            (state.builder.block_count(), state.charged_blocks)
        })
    }

    /// Charge the blocks created since `mark` that an inner pass has not charged.
    pub(super) fn charge_pass_blocks(&mut self, mark: (usize, usize), line: usize) -> Result<()> {
        let state = self.state();
        let created = state.builder.block_count() - mark.0;
        let own = created - (state.charged_blocks - mark.1);
        state.charged_blocks += own;
        self.charge(own * BLOCK_UNITS, line)
    }

    pub(super) fn is_bit_name(&self, name: &str) -> bool {
        self.cregs.contains_key(name)
            || self
                .alias(name)
                .is_some_and(|alias| alias.kind == ast::RegisterKind::Classical)
    }

    /// A classical bit or bits a name stands for, read as a value.
    fn bit_leaf(&self, name: &str, line: usize) -> Result<Option<ClassicalExpr>> {
        if let Some(register) = self.cregs.get(name) {
            return register_read(register.offset, register.size, line).map(Some);
        }
        match self.alias(name) {
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
            Expr::Runtime(var) => Ok(Some(ClassicalExpr::Var(*var))),
            Expr::Ident(name) => {
                if let Some(var) = self.runtime_var(name) {
                    return Ok(Some(ClassicalExpr::Var(var)));
                }
                self.bit_leaf(name, line)
            }
            Expr::Element(element) => {
                if let Some(var) = self.runtime_var(element.array) {
                    let Some(at) = self.int_bit(element, line)? else {
                        return Ok(None);
                    };
                    let shifted = ClassicalExpr::binary(
                        dynamic::BinaryOp::Shr,
                        ClassicalExpr::Var(var),
                        ClassicalExpr::Const(ClassicalValue::Int(at.into())),
                    );
                    return Ok(Some(ClassicalExpr::binary(
                        dynamic::BinaryOp::BitAnd,
                        shifted,
                        ClassicalExpr::Const(ClassicalValue::Int(1)),
                    )));
                }
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

    pub(super) fn lower_expr(&self, expr: &Expr, line: usize) -> Result<ClassicalExpr> {
        lower(
            expr,
            line,
            &|leaf| self.runtime_leaf(leaf, line),
            &|folded| self.typed_of(folded, line),
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

    /// The parse-time type of a `float[n]` or `angle[n]` declaration.
    pub(super) fn real_type(
        &self,
        ty: &str,
        width: Option<&Expr>,
        line: usize,
    ) -> Result<ClassicalType> {
        let float = ty != "angle";
        let bits = width
            .map(|width| {
                let bits = self.integer_of(width, line)?;
                let fits = if float {
                    matches!(bits, 16 | 32 | 64 | 128)
                } else {
                    (1..=64).contains(&bits)
                };
                let range = if float {
                    "16, 32, 64 or 128"
                } else {
                    "1 to 64"
                };
                fits.then_some(bits as u32)
                    .ok_or_else(|| parse_error(line, format!("`{ty}[{bits}]` needs {range} bits")))
            })
            .transpose()?;
        Ok(if float {
            ClassicalType::Float(bits.unwrap_or(64))
        } else {
            ClassicalType::Angle(bits)
        })
    }

    pub(super) fn runtime_type(
        &self,
        ty: &str,
        width: Option<&Expr>,
        line: usize,
    ) -> Result<RuntimeType> {
        if matches!(ty, "float" | "angle") {
            return Ok(match self.real_type(ty, width, line)? {
                ClassicalType::Float(width) => RuntimeType::Float { width },
                _ => RuntimeType::Angle,
            });
        }
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
                construct: format!("`const {name}` initializer needs a parse-time constant"),
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
            RuntimeType::Float { width } => ClassicalType::Float(width),
            RuntimeType::Angle => ClassicalType::Angle(None),
            integer => ClassicalType::Int(integer),
        };
        let decl = ClassicalDecl {
            ty: kind,
            constant: false,
            value: None,
        };
        if Rc::make_mut(&mut self.classical)
            .insert(name, decl)
            .is_none()
        {
            self.declared.push(name);
        }
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
            value = ClassicalExpr::binary(op.binary().runtime(), ClassicalExpr::Var(var), value);
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
        self.charge(width, line)?;
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
        let returned = self.def_result.as_ref().is_some_and(|frame| frame.returned);
        let result = self.execute(block);
        if let Some(frame) = self.def_result.as_mut().filter(|frame| frame.runtime) {
            frame.returned = returned;
        }
        self.state().repeat -= 1;
        self.nested = was_nested;
        self.close_scope(scope);
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
        if let Ok(expr) = condition_expr(condition)
            && expr
                .finds_call(&|name| self.def_defs.contains_key(name))
                .is_some()
        {
            self.state()
                .builder
                .begin_while_test(format!("while at line {line}"));
            let mut instructions = Vec::new();
            let value = self.resolve_calls(&expr, line, &mut instructions)?;
            self.flush(instructions);
            let value = self.lower_expr(&value, line)?;
            self.state().builder.finish_while_test(value)?;
        } else {
            let condition = self.lower_condition(condition, line)?;
            self.state()
                .builder
                .begin_while(format!("while at line {line}"), condition);
        }
        self.loops.push(Loop::While);
        let result = self.branch_body(body);
        self.loops.pop();
        result?;
        self.state().builder.end()
    }

    /// A `break` or `continue`. One every path through the innermost `for`
    /// reaches stops the unrolling there; one under a measured branch jumps in
    /// the graph.
    pub(super) fn exec_loop_exit(&mut self, exit: Exit, line: usize) -> Result<()> {
        let word = if exit == Exit::Break {
            "break"
        } else {
            "continue"
        };
        let Some(&innermost) = self.loops.last() else {
            return Err(parse_error(line, format!("`{word}` outside a loop")));
        };
        if let Loop::For {
            region_depth,
            frames,
            graph,
        } = innermost
        {
            if region_depth == self.region_depth && frames == self.graph_depth() {
                self.exit = Some(exit);
                return Ok(());
            }
            if !graph {
                return Err(self.decided_by_measurement(word, line));
            }
        }
        self.require_streaming(&format!("`{word}`"), line)?;
        let builder = &mut self.state().builder;
        if exit == Exit::Break {
            builder.break_loop()
        } else {
            builder.continue_loop()
        }
    }

    /// `end`: the program stops at parse time when every path reaches it, and
    /// at a halting block of the graph when a measurement decides that.
    pub(super) fn exec_end(&mut self, line: usize) -> Result<()> {
        if self.dynamic.is_none() {
            if self.region_depth > 0 {
                return Err(self.decided_by_measurement("end", line));
            }
            self.exit = Some(Exit::End);
            return Ok(());
        }
        self.require_streaming("`end`", line)?;
        if self.graph_depth() > 0 {
            // An `end` every path reaches, inside calls and unrolled passes no
            // path has left, stops the parse there as it does at the top level.
            if self.state().builder.on_every_path() {
                self.exit = Some(Exit::End);
            }
            self.state().builder.halt();
        } else if self.region_depth > 0 {
            return Err(self.decided_by_measurement("end", line));
        } else {
            self.exit = Some(Exit::End);
        }
        Ok(())
    }

    fn decided_by_measurement(&self, word: &str, line: usize) -> PrismError {
        let remedy = if self.dynamic.is_none() {
            "parse the program with `openqasm::parse_dynamic` to lower it into the graph"
        } else {
            "give the branch a condition over a runtime value to make it a branch"
        };
        PrismError::UnsupportedConstruct {
            construct: format!("`{word}` under a measurement-dependent branch; {remedy}"),
            line,
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
        if let Ok(expr) = condition_expr(&conditional.condition)
            && expr
                .finds_call(&|name| self.def_defs.contains_key(name))
                .is_some()
        {
            let value = self.resolve_calls(&expr, line, out)?;
            let value = self.lower_expr(&value, line)?;
            if let ClassicalExpr::Const(value) = value {
                let body = if value.truthy() {
                    Some(&conditional.then_body)
                } else {
                    conditional.else_body.as_ref()
                };
                if let Some(body) = body {
                    out.extend(self.exec_box(body)?);
                }
                return Ok(());
            }
            return self.branch_on(value, conditional, line);
        }
        let decided = self.decided(&conditional.condition);
        let needs_graph = self
            .dynamic
            .as_ref()
            .is_some_and(|state| state.analysis.branches.contains(&key(stmt)));
        if decided == Decided::Static {
            let value = self.resolve_calls(&condition_expr(&conditional.condition)?, line, out)?;
            let taken = self.value_of(&value, line)? != 0.0;
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
        let condition = self.lower_condition(&conditional.condition, line)?;
        self.branch_on(condition, conditional, line)
    }

    fn branch_on(
        &mut self,
        condition: ClassicalExpr,
        conditional: &ast::Conditional<'a>,
        line: usize,
    ) -> Result<()> {
        self.require_streaming("an `if` over a runtime value", line)?;
        self.state().builder.begin_if(condition);
        self.branch_body(&conditional.then_body)?;
        if let Some(else_body) = &conditional.else_body {
            self.state().builder.begin_else()?;
            self.branch_body(else_body)?;
        }
        self.state().builder.end()
    }

    /// Whether a `switch` lowers to branches in the graph rather than guards.
    pub(super) fn switch_needs_graph(&self, stmt: &Stmt, target: &Expr) -> bool {
        let Some(state) = self.dynamic.as_ref() else {
            return false;
        };
        let runtime_var = |leaf: &Expr| {
            Ok(match leaf {
                Expr::Ident(name) => self.runtime_var(name),
                Expr::Element(element) => self.runtime_var(element.array),
                _ => None,
            }
            .map(ClassicalExpr::Var))
        };
        state.analysis.branches.contains(&key(stmt))
            || reads_leaf(target, &runtime_var).unwrap_or(true)
            || self.lower_expr(target, 0).is_ok_and(|lowered| {
                !matches!(lowered, ClassicalExpr::Const(_))
                    && super::eval::register_value(&lowered).is_none()
            })
    }

    /// A `switch` as a chain of branches, one per arm, the `default` in the
    /// last `else`.
    pub(super) fn exec_switch_branches(
        &mut self,
        target: &Expr<'a>,
        arms: &[ast::SwitchArm<'a>],
        line: usize,
    ) -> Result<()> {
        self.require_streaming("a `switch` over a runtime value", line)?;
        let value = self.lower_expr(target, line)?;
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
    /// `for` loops a measurement may break or continue.
    unrolled: HashSet<usize>,
    runtime_defs: Rc<HashSet<String>>,
    needs_call: bool,
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
    functions: &'s HashSet<String>,
    scalar_return: bool,
    needs_call: bool,
    local_bits: HashSet<&'a str>,
    initialized_bits: HashSet<&'a str>,
    single_bits: HashSet<&'a str>,
    early_bit_read: Cell<bool>,
    conditional: usize,
    /// Runtime depth of each enclosing loop body, innermost last.
    loops: Vec<usize>,
    /// `for` loops found to hold a measured exit, kept across passes of the
    /// fixed point so each body is walked once per pass.
    unrolled: &'s mut HashSet<usize>,
    /// The innermost `for` holds a `break` or `continue` a measurement decides.
    runtime_exit: bool,
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
        let mut functions = Rc::new(HashSet::new());
        loop {
            let mut changed = false;
            for stmt in program {
                if let StmtKind::DefDef {
                    name,
                    args,
                    result,
                    body,
                } = &stmt.kind
                    && !functions.contains(*name)
                {
                    let needs_call = Self::for_def(
                        body,
                        args,
                        result.as_ref().is_some_and(|result| result.ty != "bit"),
                        Rc::clone(&functions),
                        &[],
                    )
                    .needs_call;
                    if needs_call {
                        Rc::make_mut(&mut functions).insert((*name).to_string());
                        changed = true;
                    }
                }
            }
            if !changed {
                return Self::for_def(program, &[], false, functions, &[]);
            }
        }
    }

    pub(super) fn for_def<'a>(
        program: &[Stmt<'a>],
        args: &[ast::DefParam<'a>],
        scalar_return: bool,
        functions: Rc<HashSet<String>>,
        runtime_params: &[&str],
    ) -> RuntimeAnalysis {
        let mut runtime = HashSet::new();
        let mut unrolled = HashSet::new();
        loop {
            let mut walk = Walk {
                runtime: &mut runtime,
                branches: HashSet::new(),
                changed: false,
                scopes: vec![HashMap::new()],
                functions: &functions,
                scalar_return,
                needs_call: false,
                local_bits: HashSet::new(),
                initialized_bits: HashSet::new(),
                single_bits: HashSet::new(),
                early_bit_read: Cell::new(false),
                conditional: 0,
                loops: Vec::new(),
                unrolled: &mut unrolled,
                runtime_exit: false,
            };
            for param in args {
                match param {
                    ast::DefParam::Bit { name, .. } => walk.bind(name, Binding::Bits),
                    ast::DefParam::Value { name, .. } => {
                        let decl = param as *const ast::DefParam as usize;
                        if runtime_params.contains(name) {
                            walk.mark(decl);
                        }
                        walk.bind(name, Binding::Var { decl, depth: 0 });
                    }
                    ast::DefParam::Duration(name) | ast::DefParam::Qubit { name, .. } => {
                        walk.bind(name, Binding::Constant);
                    }
                }
            }
            walk.block(program, 0);
            if !walk.changed {
                let branches = walk.branches;
                let initializes_bits = walk.early_bit_read.get();
                let needs_call = walk.needs_call || initializes_bits || !runtime.is_empty();
                return RuntimeAnalysis {
                    runtime,
                    branches,
                    unrolled,
                    runtime_defs: functions,
                    needs_call,
                };
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
            Binding::Bits => {
                if self.local_bits.contains(name) && !self.initialized_bits.contains(name) {
                    self.early_bit_read.set(true);
                }
                Reads {
                    runtime: false,
                    bits: true,
                }
            }
            Binding::Constant | Binding::Other => Reads::default(),
        }
    }

    fn expr_reads(&self, expr: &Expr) -> Reads {
        with_stack(|| match expr {
            Expr::Runtime(_) => Reads {
                runtime: true,
                bits: false,
            },
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
            Expr::Call(call) => call.args.iter().fold(
                Reads {
                    runtime: self.functions.contains(call.name),
                    bits: false,
                },
                |reads, arg| reads.or(self.expr_reads(arg)),
            ),
            Expr::Cast(cast) => self.expr_reads(&cast.value),
        })
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
        with_stack(|| {
            let mut needs = false;
            for stmt in block {
                needs |= self.stmt(stmt, depth);
            }
            needs
        })
    }

    fn stmt(&mut self, stmt: &Stmt<'a>, depth: usize) -> bool {
        #[cfg(test)]
        WALKED.with(|walked| walked.set(walked.get() + 1));
        match &stmt.kind {
            StmtKind::RegisterDecl {
                kind,
                name,
                size,
                init,
            } => {
                let binding = match kind {
                    ast::RegisterKind::Classical => Binding::Bits,
                    ast::RegisterKind::Qubit => Binding::Other,
                };
                self.bind(name, binding);
                if *kind == ast::RegisterKind::Classical {
                    self.local_bits.insert(name);
                    if size.is_none() || matches!(size, Some(Expr::Number(ClassicalValue::Int(1))))
                    {
                        self.single_bits.insert(name);
                    }
                    self.needs_call |= init.is_some();
                }
                match init.as_deref() {
                    Some(ast::BitInit::Bits(text)) => text.contains('1'),
                    Some(ast::BitInit::Value(value)) => !matches!(
                        value,
                        Expr::Number(
                            dynamic::ClassicalValue::Int(0) | dynamic::ClassicalValue::Bool(false)
                        )
                    ),
                    None => false,
                }
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
            StmtKind::Assign { target, value, .. } => match self.lookup(target) {
                Binding::Bits => {
                    self.needs_call = true;
                    true
                }
                binding => {
                    let reads = self.expr_reads(value);
                    self.write(binding, depth, reads)
                }
            },
            StmtKind::Measure(measure) => {
                if depth == 0
                    && self.conditional == 0
                    && let Some(name) = measure.target.register()
                    && self.local_bits.contains(name)
                    && (measure.target.index.is_none() || self.single_bits.contains(name))
                {
                    self.initialized_bits.insert(name);
                }
                false
            }
            StmtKind::CallAssign(assign) if assign.target.index.is_none() => {
                let Some(target) = assign.target.register() else {
                    return false;
                };
                let dynamic_call = self.functions.contains(assign.name);
                self.needs_call |= dynamic_call;
                let reads = assign.args.iter().fold(
                    Reads {
                        runtime: dynamic_call,
                        bits: false,
                    },
                    |reads, arg| reads.or(self.argument_reads(arg)),
                );
                self.write(self.lookup(target), depth, reads) || dynamic_call
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
            StmtKind::ElementAssign(assign)
                if matches!(self.lookup(assign.array), Binding::Bits) =>
            {
                self.needs_call = true;
                true
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
            StmtKind::Call { name, params, .. } => {
                let dynamic_call = self.functions.contains(*name);
                self.needs_call |= dynamic_call;
                dynamic_call || params.iter().any(|param| self.argument_reads(param).any())
            }
            StmtKind::Expr(value) => {
                let runtime = self.expr_reads(value).runtime;
                self.needs_call |= runtime;
                runtime
            }
            StmtKind::Return(value) => {
                let reads = match value {
                    Some(ast::ReturnValue::Value(value)) => self.expr_reads(value),
                    _ => Reads::default(),
                };
                self.needs_call |= depth > 0
                    || self.scalar_return && reads.any()
                    || !self.scalar_return
                        && matches!(value, Some(ast::ReturnValue::Value(value))
                        if value.as_ident().is_none_or(|name| !self.local_bits.contains(name)));
                true
            }
            StmtKind::If(conditional) => {
                self.conditional += 1;
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
                self.conditional -= 1;
                needs
            }
            StmtKind::Switch { target, arms } => {
                self.conditional += 1;
                let reads = self.expr_reads(target);
                let mut needs = reads.runtime;
                for arm in arms {
                    needs |= self.scoped(&arm.body, depth + usize::from(reads.any()));
                }
                if needs {
                    self.branches.insert(key(stmt));
                }
                self.conditional -= 1;
                needs
            }
            StmtKind::While { body, .. } => {
                self.needs_call = true;
                self.loops.push(depth + 1);
                let outer_exit = std::mem::replace(&mut self.runtime_exit, false);
                self.scoped(body, depth + 1);
                self.runtime_exit = outer_exit;
                self.loops.pop();
                true
            }
            StmtKind::Break | StmtKind::Continue => {
                let runtime = self.loops.last().is_some_and(|&home| depth > home);
                self.runtime_exit |= runtime;
                runtime
            }
            StmtKind::End => {
                // A def that may end the program runs as a call, so its `end`
                // halts the shot wherever a measurement decides the call.
                self.needs_call = true;
                depth > 0
            }
            StmtKind::For { variable, body, .. } => {
                self.conditional += 1;
                // A pass a measurement may cut short leaves every write in the
                // body, and every pass after it, decided at runtime. A loop
                // found to hold such an exit is walked one level down from the
                // next pass of the fixed point on.
                let unrolled = self.unrolled.contains(&key(stmt));
                let inner = depth + usize::from(unrolled);
                self.loops.push(inner);
                let outer_exit = std::mem::replace(&mut self.runtime_exit, false);
                self.scopes.push(HashMap::new());
                self.bind(variable, Binding::Constant);
                let mut needs = self.block(body, inner);
                self.scopes.pop();
                if self.runtime_exit {
                    if !unrolled {
                        self.unrolled.insert(key(stmt));
                        self.changed = true;
                    }
                    self.needs_call = true;
                    needs = true;
                }
                self.runtime_exit = outer_exit;
                self.loops.pop();
                self.conditional -= 1;
                needs
            }
            StmtKind::Box { body, .. } => self.scoped(body, depth),
            _ => false,
        }
    }
}
