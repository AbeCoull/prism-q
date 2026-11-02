//! Walking a parsed program into instructions.
//!
//! Every name a statement reads is resolved against the scope the evaluator
//! holds, so a `for` body or a `gate` body is parsed once and run against
//! fresh bindings rather than rewritten and reparsed.

use std::collections::HashMap;

use super::*;
use crate::circuit::dynamic::ClassicalExpr;
use crate::circuit::qasm::ast::{
    self, Argument, AssignOp, Block, CmpOp, Condition, DefParam, ForRange, Index, Operand,
    OperandName, Stmt, StmtKind,
};
use crate::circuit::qasm::expr::{self as syntax_expr, Duration, Expr, Timed};
use crate::circuit::qasm::stream::{MAX_NESTING, with_stack};
use crate::circuit::qasm::{lexer, parser as syntax};

/// Register width the highest `$k` the program names implies, `None` when it
/// names none.
///
/// Physical qubits declare no register, so the width comes from the tree
/// before any of it runs.
pub(super) fn physical_width(block: &Block) -> Result<Option<usize>> {
    let mut highest: Option<(usize, usize)> = None;
    let mut note = |operand: &Operand| {
        if let OperandName::Physical(index) = operand.name
            && highest.is_none_or(|(seen, _)| seen < index)
        {
            highest = Some((index, operand.line));
        }
    };
    walk_operands(block, &mut note);
    match highest {
        None => Ok(None),
        Some((index, line)) if index >= MAX_DECLARED_WIDTH => {
            Err(too_wide(ast::RegisterKind::Qubit, line))
        }
        Some((index, _)) => Ok(Some(index + 1)),
    }
}

fn too_wide(kind: ast::RegisterKind, line: usize) -> PrismError {
    parse_error(
        line,
        format!(
            "the program declares more than {MAX_DECLARED_WIDTH} {}s",
            kind.name()
        ),
    )
}

fn walk_operands(block: &Block, note: &mut impl FnMut(&Operand)) {
    with_stack(|| {
        for stmt in block {
            match &stmt.kind {
                StmtKind::Alias { sources, .. } => sources.iter().for_each(&mut *note),
                StmtKind::Call {
                    params, operands, ..
                } => {
                    for param in params {
                        if let Argument::Operand(operand) = param {
                            note(operand);
                        }
                    }
                    operands.iter().for_each(&mut *note);
                }
                StmtKind::Measure(measure) => {
                    note(&measure.source);
                    note(&measure.target);
                }
                StmtKind::CallAssign(assign) => {
                    note(&assign.target);
                    for arg in &assign.args {
                        if let Argument::Operand(operand) = arg {
                            note(operand);
                        }
                    }
                }
                StmtKind::Return(Some(ast::ReturnValue::Measure(source))) => note(source),
                StmtKind::Reset { targets }
                | StmtKind::Barrier { targets }
                | StmtKind::Delay { targets, .. }
                | StmtKind::Nop { targets } => targets.iter().for_each(&mut *note),
                StmtKind::If(conditional) => {
                    match &conditional.condition {
                        Condition::Truthy(operand) | Condition::Negated(operand) => note(operand),
                        Condition::Compare { lhs, .. } => note(lhs),
                        Condition::Parity { bits, .. } => bits.iter().for_each(&mut *note),
                        Condition::Expr(_) => {}
                    }
                    walk_operands(&conditional.then_body, note);
                    if let Some(body) = &conditional.else_body {
                        walk_operands(body, note);
                    }
                }
                StmtKind::For { body, .. }
                | StmtKind::While { body, .. }
                | StmtKind::Box { body, .. }
                | StmtKind::DefDef { body, .. }
                | StmtKind::GateDef { body, .. } => walk_operands(body, note),
                StmtKind::Switch { arms, .. } => {
                    for arm in arms {
                        walk_operands(&arm.body, note);
                    }
                }
                _ => {}
            }
        }
    })
}

/// Fold an expression to a number, where a ratio of two durations counts as
/// one. The duration path runs only once the plain fold has failed, so a
/// program that names no duration never pays for it.
fn number_of(
    expr: &Expr,
    line: usize,
    vars: Option<&HashMap<&str, ClassicalValue>>,
    durations: &HashMap<&str, Timed>,
    arrays: &HashMap<&str, Rc<ClassicalArray>>,
) -> Result<ClassicalValue> {
    let value = syntax_expr::eval(expr, line, vars);
    if value.is_ok() {
        return value;
    }
    if !arrays.is_empty() && expr.reads_element() {
        let resolved = resolve_elements(expr, line, vars, durations, arrays)?;
        return number_of(&resolved, line, vars, durations, arrays);
    }
    if !syntax_expr::is_timed(expr, durations) {
        return value;
    }
    match syntax_expr::eval_timed(expr, line, vars, durations)? {
        Timed::Number(value) => Ok(ClassicalValue::Float(value)),
        Timed::Duration(_) => Err(parse_error(
            line,
            format!("`{expr}` is a duration where a number belongs"),
        )),
        Timed::Stretch => Err(parse_error(
            line,
            format!("`{expr}` is a stretch where a number belongs"),
        )),
    }
}

/// Fold a timing expression, resolving the array elements it reads first.
fn timed_of(
    expr: &Expr,
    line: usize,
    vars: Option<&HashMap<&str, ClassicalValue>>,
    durations: &HashMap<&str, Timed>,
    arrays: &HashMap<&str, Rc<ClassicalArray>>,
) -> Result<Timed> {
    if arrays.is_empty() || !expr.reads_element() {
        return syntax_expr::eval_timed(expr, line, vars, durations);
    }
    let resolved = resolve_elements(expr, line, vars, durations, arrays)?;
    syntax_expr::eval_timed(&resolved, line, vars, durations)
}

/// `expr` with each array element it reads replaced by that element's value.
fn resolve_elements<'e>(
    expr: &Expr<'e>,
    line: usize,
    vars: Option<&HashMap<&str, ClassicalValue>>,
    durations: &HashMap<&str, Timed>,
    arrays: &HashMap<&str, Rc<ClassicalArray>>,
) -> Result<Expr<'e>> {
    replace_elements(expr, &|element| {
        let array = arrays.get(element.array).ok_or_else(|| {
            parse_error(line, format!("`{}` is not a declared array", element.array))
        })?;
        let at = element_offset(array, element.array, &element.indices, line, |index| {
            with_stack(|| number_of(index, line, vars, durations, arrays))
        })?;
        Ok(Some(array.values[at]))
    })
}

/// `expr` with each element `value` gives a value for replaced by it.
fn replace_elements<'e>(
    expr: &Expr<'e>,
    value: &impl Fn(&syntax_expr::Element<'e>) -> Result<Option<ClassicalValue>>,
) -> Result<Expr<'e>> {
    let resolve = |inner: &Expr<'e>| with_stack(|| replace_elements(inner, value));
    Ok(match expr {
        Expr::Element(element) => match value(element)? {
            Some(value) => Expr::Number(value),
            None => expr.clone(),
        },
        Expr::Negate(inner) => Expr::Negate(Box::new(resolve(inner)?)),
        Expr::Not(inner) => Expr::Not(Box::new(resolve(inner)?)),
        Expr::BitNot(inner) => Expr::BitNot(Box::new(resolve(inner)?)),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(resolve(left)?),
            right: Box::new(resolve(right)?),
        },
        Expr::Call(call) => Expr::Call(Box::new(syntax_expr::Call {
            name: call.name,
            args: call.args.iter().map(resolve).collect::<Result<_>>()?,
        })),
        Expr::Cast(cast) => Expr::Cast(Box::new(syntax_expr::Cast {
            ty: cast.ty,
            width: cast.width.as_ref().map(resolve).transpose()?,
            value: resolve(&cast.value)?,
        })),
        other => other.clone(),
    })
}

/// Row-major position of `array[indices]`, each index folded by `fold`.
fn element_offset(
    array: &ClassicalArray,
    name: &str,
    indices: &[Expr],
    line: usize,
    fold: impl Fn(&Expr) -> Result<ClassicalValue>,
) -> Result<usize> {
    if indices.len() != array.dims.len() {
        return Err(parse_error(
            line,
            format!(
                "`{name}` has {} dimension(s) but is indexed with {}",
                array.dims.len(),
                indices.len()
            ),
        ));
    }
    let mut at = 0usize;
    for (index, &size) in indices.iter().zip(&array.dims) {
        let written = fold(index)?.to_float();
        let value = if written < 0.0 {
            written + size as f64
        } else {
            written
        };
        if value.fract() != 0.0 || value < 0.0 || value >= size as f64 {
            return Err(parse_error(
                line,
                format!("index {written} is outside `{name}`, whose dimension has size {size}"),
            ));
        }
        at = at * size + value as usize;
    }
    Ok(at)
}

/// The guard a lowered condition spells, when it has one: a bit or its negation,
/// or a register, a cast of one, or a bit compared with a whole number.
fn guard_condition(condition: &ClassicalExpr) -> Option<ClassicalCondition> {
    use crate::circuit::dynamic::{BinaryOp as Op, ClassicalType as Ty, UnaryOp};
    match condition {
        ClassicalExpr::Bit(bit) => Some(ClassicalCondition::BitIsOne(*bit)),
        ClassicalExpr::Register { offset, size } => Some(ClassicalCondition::RegisterNotEquals {
            offset: *offset,
            size: *size,
            value: 0,
        }),
        ClassicalExpr::Unary {
            op: UnaryOp::Not,
            operand,
        } => guard_condition(operand).map(|guard| guard.negate()),
        ClassicalExpr::Cast {
            ty: Ty::Bool,
            operand,
        } => guard_condition(operand),
        ClassicalExpr::Binary {
            op: op @ (Op::Eq | Op::Ne),
            lhs,
            rhs,
        } => {
            let (bits, value) = match (lhs.as_ref(), rhs.as_ref()) {
                (ClassicalExpr::Const(value), bits) | (bits, ClassicalExpr::Const(value)) => {
                    (bits, *value)
                }
                _ => return None,
            };
            if matches!(value, ClassicalValue::Float(float) if float.fract() != 0.0) {
                return None;
            }
            let (span, value) = match bits {
                ClassicalExpr::Cast {
                    ty: Ty::Int { width },
                    operand,
                } => {
                    let (offset, size) = register_value(operand)?;
                    let width = *width as usize;
                    if width > size {
                        ((offset, size), u64::try_from(value.to_int()).ok()?)
                    } else {
                        ((offset, width), signed_pattern(value.to_int(), width)?)
                    }
                }
                _ => (register_value(bits)?, u64::try_from(value.to_int()).ok()?),
            };
            let equals = match span {
                (bit, 1) if value <= 1 => {
                    if value == 1 {
                        ClassicalCondition::BitIsOne(bit)
                    } else {
                        ClassicalCondition::BitIsZero(bit)
                    }
                }
                (offset, size) => ClassicalCondition::RegisterEquals {
                    offset,
                    size,
                    value,
                },
            };
            Some(if *op == Op::Eq {
                equals
            } else {
                equals.negate()
            })
        }
        _ => None,
    }
}

/// The low `width` bits an `int[width]` holding `value` spells, when it can.
fn signed_pattern(value: i128, width: usize) -> Option<u64> {
    let half = 1i128 << (width - 1);
    (-half..half)
        .contains(&value)
        .then(|| (value & ((half << 1) - 1)) as u64)
}

/// The span of bits `expr` reads as an unsigned value: a bit, a register, or an
/// unsigned cast of one, which keeps the low bits a narrower width names. A
/// signed cast reads as unsigned only when wider than its bits.
pub(super) fn register_value(expr: &ClassicalExpr) -> Option<(usize, usize)> {
    use crate::circuit::dynamic::ClassicalType as Ty;
    match expr {
        ClassicalExpr::Bit(bit) => Some((*bit, 1)),
        ClassicalExpr::Register { offset, size } => Some((*offset, *size)),
        ClassicalExpr::Cast { ty, operand } => {
            let (offset, size) = register_value(operand)?;
            match *ty {
                Ty::Uint { width } => Some((offset, size.min(width as usize))),
                Ty::Int { width } if (width as usize) > size => Some((offset, size)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_bit_param(args: &[DefParam], name: &str) -> bool {
    args.iter()
        .any(|arg| matches!(arg, DefParam::Bit { name: param, .. } if *param == name))
}

/// Line of the first `return` in `block`, nested bodies included.
fn return_line(block: &[Stmt]) -> Option<usize> {
    with_stack(|| {
        block.iter().find_map(|stmt| match &stmt.kind {
            StmtKind::Return(_) => Some(stmt.line),
            StmtKind::If(conditional) => return_line(&conditional.then_body)
                .or_else(|| conditional.else_body.as_deref().and_then(return_line)),
            StmtKind::For { body, .. }
            | StmtKind::While { body, .. }
            | StmtKind::Box { body, .. } => return_line(body),
            StmtKind::Switch { arms, .. } => arms.iter().find_map(|arm| return_line(&arm.body)),
            _ => None,
        })
    })
}

impl<'a> Parser<'a> {
    /// Lex, parse, and run the source.
    pub(super) fn parse_program(
        &mut self,
    ) -> Result<(Circuit, Parameters, Vec<ResultSpec>, Option<NoiseModel>)> {
        let tokens = lexer::tokenize(self.input)?;
        let program = syntax::parse_program(&tokens)?;
        if let Some(width) = physical_width(&program)? {
            self.physical = true;
            self.total_qubits = width;
        }
        let instructions = self.execute(&program)?;
        if self.verbatim_pending {
            return Err(parse_error(
                program.last().map_or(1, |stmt| stmt.line),
                "a verbatim pragma must be followed by a `box`",
            ));
        }

        let circuit = Circuit {
            num_qubits: self.total_qubits,
            num_classical_bits: self.total_cbits,
            instructions,
        };
        let params =
            Parameters::from_links(std::mem::take(&mut self.links), self.input_names.len())
                .with_names(std::mem::take(&mut self.input_names))
                .pinned_to(&circuit);
        let noise = self.build_noise_model(&circuit)?;
        Ok((circuit, params, std::mem::take(&mut self.results), noise))
    }

    /// Run a block, threading the parameter link and noise event each statement
    /// may leave behind onto the instruction index they belong to.
    pub(super) fn execute(&mut self, block: &Block<'a>) -> Result<Vec<Instruction>> {
        if self.exec_depth >= MAX_NESTING {
            return Err(parse_error(
                block.first().map_or(1, |stmt| stmt.line),
                format!(
                    "the program nests deeper than {MAX_NESTING} levels, counting the \
                     gate and def bodies it expands"
                ),
            ));
        }
        self.exec_depth += 1;
        let result = with_stack(|| self.execute_block(block));
        self.exec_depth -= 1;
        result
    }

    fn execute_block(&mut self, block: &Block<'a>) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::with_capacity(block.len());
        for stmt in block {
            let base = instructions.len();
            let left = self.expansion_left.get();
            self.pending_input_slot = None;
            self.exec_stmt(stmt, &mut instructions)?;
            // Nested blocks charged what they emitted as they ran; this
            // statement owes only what it added on top.
            let nested = left - self.expansion_left.get();
            self.charge(
                (instructions.len() - base).saturating_sub(nested).max(1),
                stmt.line,
            )?;
            if let Some(slot) = self.pending_input_slot.take() {
                for offset in 0..instructions.len() - base {
                    self.links.push(ParamLink {
                        instruction: base + offset,
                        slot,
                    });
                }
            }
            if let Some(spec) = self.pending_noise.take() {
                // A channel acts at the point the pragma stands, which is after
                // everything emitted so far.
                if base == 0 {
                    return Err(parse_error(
                        stmt.line,
                        "a noise pragma needs a preceding instruction to follow",
                    ));
                }
                self.noise_specs.push((base - 1, spec));
            }
            if self.streaming() {
                let produced = std::mem::take(&mut instructions);
                self.flush(produced);
            }
        }
        Ok(instructions)
    }

    pub(super) fn charge(&self, units: usize, line: usize) -> Result<()> {
        let left = self.expansion_left.get();
        self.expansion_left.set(left.saturating_sub(units));
        if units > left {
            return Err(Self::expansion_exhausted(line));
        }
        Ok(())
    }

    pub(super) fn expansion_exhausted(line: usize) -> PrismError {
        parse_error(
            line,
            format!(
                "the program expands past {MAX_EXPANSION} instructions and statements, \
                 counting every gate call, loop pass and repetition"
            ),
        )
    }

    fn exec_stmt(&mut self, stmt: &Stmt<'a>, out: &mut Vec<Instruction>) -> Result<()> {
        let line = stmt.line;
        match &stmt.kind {
            StmtKind::Empty | StmtKind::Include => Ok(()),
            StmtKind::Expr(value) => {
                match value.finds_call(&|name| self.def_defs.contains_key(name)) {
                    Some(def) => Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "a call to def `{def}` inside an expression, whose result \
                             nothing yet carries"
                        ),
                        line,
                    }),
                    None => Ok(()),
                }
            }
            StmtKind::Version(version) => {
                Self::check_version_number(version, line)?;
                Ok(())
            }
            StmtKind::RegisterDecl {
                kind,
                name,
                size,
                init,
            } => {
                if let Some(result) = self.def_result.as_ref().filter(|result| {
                    *kind == ast::RegisterKind::Classical && result.local == Some(*name)
                }) {
                    if init.is_some() {
                        return Err(PrismError::UnsupportedConstruct {
                            construct: format!(
                                "an initializer on `{name}`, the bit def returns; the \
                                 caller's bits hold it and the instruction list has no \
                                 classical write"
                            ),
                            line,
                        });
                    }
                    // The returned bit is the caller's target, bound before the
                    // body ran, so its declaration allocates nothing.
                    let width = match size {
                        Some(size) => self.integer_of(size, line)?,
                        None => 1,
                    };
                    if width != result.bits.len() as i64 {
                        return Err(parse_error(
                            line,
                            format!(
                                "`{name}` declares {width} bit(s) where the def returns {}",
                                result.bits.len()
                            ),
                        ));
                    }
                    return Ok(());
                }
                if self.in_while() {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "register `{name}` declared inside a `while` loop; declare it \
                             before the loop"
                        ),
                        line,
                    });
                }
                self.declare_register(*kind, name, size.as_ref(), line)?;
                if let Some(init) = init {
                    self.init_bits(name, size.as_ref(), init, line)?;
                }
                Ok(())
            }
            StmtKind::InputDecl { ty, name } => {
                if self.dynamic.is_some() {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("input `{name}` in a dynamic program"),
                        line,
                    });
                }
                self.declare_input(ty, name, line)?;
                Ok(())
            }
            StmtKind::OutputDecl { ty, name, size } => {
                if *ty != "bit" {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("`output {ty}` (only bit outputs are reported)"),
                        line,
                    });
                }
                self.declare_register(ast::RegisterKind::Classical, name, size.as_ref(), line)?;
                Ok(())
            }
            StmtKind::ClassicalDecl {
                constant,
                ty,
                width,
                name,
                value,
            } => {
                if self.is_runtime_decl(stmt) {
                    return self.declare_runtime(
                        *constant,
                        ty,
                        width.as_ref(),
                        name,
                        value.as_ref(),
                        line,
                    );
                }
                self.declare_classical(*constant, ty, width.as_ref(), name, value.as_ref(), line)?;
                Ok(())
            }
            StmtKind::Assign { target, op, value } if self.is_bit_name(target) => {
                let target = Operand {
                    name: OperandName::Register(target),
                    index: None,
                    line,
                };
                self.assign_bits(&target, *op, value, line)
            }
            StmtKind::Assign { target, op, value } => {
                self.assign_classical(target, *op, value, line)?;
                Ok(())
            }
            StmtKind::Alias { name, sources } => {
                self.declare_alias(name, sources, line)?;
                Ok(())
            }
            StmtKind::Call {
                modifiers,
                name,
                params,
                operands,
            } => self.exec_call(modifiers, name, params, operands, line, out),
            StmtKind::Measure(measure) => {
                if let Some(name) = measure
                    .target
                    .register()
                    .filter(|name| self.dynamic.is_some() && self.classical.contains_key(name))
                {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "a measurement into variable `{name}`; measure into a bit and \
                             assign the bit"
                        ),
                        line,
                    });
                }
                let qubits = self.qubits_of(&measure.source)?;
                let bits = self.bits_of(&measure.target)?;
                out.extend(Self::build_measurements(qubits, bits, line)?);
                Ok(())
            }
            StmtKind::Reset { targets } => {
                for target in targets {
                    out.extend(
                        self.qubits_of(target)?
                            .into_iter()
                            .map(|qubit| Instruction::Reset { qubit }),
                    );
                }
                Ok(())
            }
            StmtKind::Barrier { targets } => {
                let mut qubits = SmallVec::<[usize; 4]>::new();
                if targets.is_empty() {
                    qubits.extend(0..self.total_qubits);
                } else {
                    for target in targets {
                        qubits.extend(self.qubits_of(target)?);
                    }
                }
                out.push(Instruction::Barrier { qubits });
                Ok(())
            }
            // An ideal simulation has no clock, so a delay is the identity on
            // its qubits. It emits nothing, which also leaves fusion free to
            // merge across it.
            StmtKind::Delay { duration, targets } => {
                self.span_of(duration, "delay", line)?;
                for target in targets {
                    self.qubits_of(target)?;
                }
                Ok(())
            }
            StmtKind::Nop { targets } => {
                for target in targets {
                    self.qubits_of(target)?;
                }
                Ok(())
            }
            StmtKind::If(conditional) if self.dynamic.is_some() => {
                self.exec_if_dynamic(stmt, conditional, line, out)
            }
            StmtKind::If(conditional) => {
                out.extend(self.exec_if(conditional, line)?);
                Ok(())
            }
            StmtKind::For {
                variable,
                range,
                body,
            } => {
                out.extend(self.exec_for(variable, range, body, line)?);
                Ok(())
            }
            StmtKind::Switch { target, arms } if self.switch_needs_graph(stmt, target) => {
                self.exec_switch_branches(target, arms, line)
            }
            StmtKind::Switch { target, arms } => {
                out.extend(self.exec_switch(target, arms, line)?);
                Ok(())
            }
            StmtKind::While { condition, body } if self.dynamic.is_some() => {
                self.exec_while(condition, body, line)
            }
            StmtKind::Break | StmtKind::Continue if self.dynamic.is_some() => {
                self.exec_loop_exit(matches!(stmt.kind, StmtKind::Break), line)
            }
            StmtKind::While { .. } | StmtKind::Break | StmtKind::Continue => {
                let construct = match stmt.kind {
                    StmtKind::While { .. } => "while",
                    StmtKind::Break => "break",
                    _ => "continue",
                };
                Err(PrismError::UnsupportedConstruct {
                    construct: construct.to_string(),
                    line,
                })
            }
            StmtKind::GateDef {
                name,
                params,
                qubits,
                body,
            } => {
                Rc::make_mut(&mut self.gate_defs).insert(
                    *name,
                    GateDefinition {
                        params: params.clone(),
                        qubits: qubits.clone(),
                        body: body.clone(),
                    },
                );
                Ok(())
            }
            StmtKind::DefDef {
                name,
                args,
                result,
                body,
            } => {
                self.declare_def(name, args, result.as_ref(), body, line)?;
                Ok(())
            }
            StmtKind::Return(value) => {
                let Some(result) = &self.def_result else {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: "`return` outside a def".to_string(),
                        line,
                    });
                };
                if let Some(ast::ReturnValue::Measure(source)) = value {
                    let qubits = self.qubits_of(source)?;
                    out.extend(Self::build_measurements(qubits, result.bits.clone(), line)?);
                }
                Ok(())
            }
            StmtKind::CallAssign(assign) => with_stack(|| self.exec_call_assign(assign, line, out)),
            StmtKind::ArrayDecl(decl) if self.is_runtime_decl(stmt) => {
                Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "array `{}`, written where a loop or a measurement decides the value",
                        decl.name
                    ),
                    line,
                })
            }
            StmtKind::ArrayDecl(decl) => self.declare_array(decl, line),
            StmtKind::ElementAssign(assign) if self.is_bit_name(assign.array) => {
                let [index] = assign.indices.as_slice() else {
                    return Err(parse_error(
                        line,
                        format!("`{}` holds bits and takes one index", assign.array),
                    ));
                };
                let target = Operand {
                    name: OperandName::Register(assign.array),
                    index: Some(Index::Single(index.clone())),
                    line,
                };
                self.assign_bits(&target, assign.op, &assign.value, line)
            }
            StmtKind::ElementAssign(assign) => self.assign_element(assign, line),
            StmtKind::Box { duration, body } => {
                // A box and Braket's verbatim pragma both direct a scheduler or
                // a device compiler, which a simulator has nothing to honour.
                self.verbatim_pending = false;
                if let Some(duration) = duration {
                    self.span_of(duration, "box", line)?;
                }
                out.extend(self.exec_box(body)?);
                Ok(())
            }
            StmtKind::Pragma(text) => {
                out.extend(self.exec_pragma(text, line)?);
                Ok(())
            }
        }
    }

    // ---------------------------------------------------------- declarations

    fn declare_register(
        &mut self,
        kind: ast::RegisterKind,
        name: &'a str,
        size: Option<&Expr<'a>>,
        line: usize,
    ) -> Result<()> {
        let width = match size {
            None => 1usize,
            Some(expr) => {
                let value = self.integer_of(expr, line)?;
                if value <= 0 {
                    return Err(parse_error(
                        line,
                        format!("{} count must be > 0", kind.name()),
                    ));
                }
                value as usize
            }
        };
        self.reject_redeclaration(name, line)?;
        let grown = |total: usize| {
            total
                .checked_add(width)
                .filter(|&total| total <= MAX_DECLARED_WIDTH)
                .ok_or_else(|| too_wide(kind, line))
        };
        match kind {
            ast::RegisterKind::Qubit => {
                self.reject_mixed_addressing(line)?;
                let offset = self.total_qubits;
                self.total_qubits = grown(offset)?;
                Rc::make_mut(&mut self.qregs).insert(
                    name,
                    Register {
                        offset,
                        size: width,
                    },
                );
            }
            ast::RegisterKind::Classical => {
                let offset = self.total_cbits;
                self.total_cbits = grown(offset)?;
                Rc::make_mut(&mut self.cregs).insert(
                    name,
                    Register {
                        offset,
                        size: width,
                    },
                );
            }
        }
        Ok(())
    }

    fn declare_input(&mut self, ty: &str, name: &'a str, line: usize) -> Result<()> {
        if !matches!(ty, "float" | "angle") {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("`input {ty}` (only float and angle inputs bind to an angle)"),
                line,
            });
        }
        if self.inputs.contains_key(name) {
            return Err(parse_error(line, format!("input `{name}` declared twice")));
        }
        self.inputs.insert(name, self.input_names.len());
        self.input_names.push(name.to_string());
        Ok(())
    }

    fn declare_classical(
        &mut self,
        constant: bool,
        ty: &str,
        width: Option<&Expr<'a>>,
        name: &'a str,
        value: Option<&Expr<'a>>,
        line: usize,
    ) -> Result<()> {
        let kind = match ty {
            "int" | "uint" => ClassicalType::Int(self.runtime_type(ty, width, line)?),
            "bool" => ClassicalType::Bool,
            "float" | "angle" => self.real_type(ty, width, line)?,
            "duration" => ClassicalType::Duration,
            "stretch" => ClassicalType::Stretch,
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`{other}` declarations"),
                    line,
                });
            }
        };
        self.reject_redeclaration(name, line)?;
        let Some(value) = value else {
            if constant {
                return Err(parse_error(
                    line,
                    format!("`const {ty} {name}` needs a value"),
                ));
            }
            if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
                let zero = Timed::Duration(Duration::default());
                self.bind_duration(name, kind, zero, constant);
            } else {
                self.bind_classical(name, kind, ClassicalValue::Int(0), constant);
            }
            return Ok(());
        };
        if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
            let folded = self.duration_of(value, line)?;
            self.bind_duration(name, kind, folded, constant);
            return Ok(());
        }
        let folded = self.fold_typed(kind, value, line)?;
        self.bind_classical(name, kind, folded, constant);
        Ok(())
    }

    /// Bind a `duration` or `stretch`. A stretch keeps no value: a scheduler
    /// sizes it, so whatever it was initialised with is only a lower bound.
    fn bind_duration(&mut self, name: &'a str, ty: ClassicalType, value: Timed, constant: bool) {
        let value = if ty == ClassicalType::Stretch {
            Timed::Stretch
        } else {
            value
        };
        let decl = ClassicalDecl { ty, constant };
        if Rc::make_mut(&mut self.classical)
            .insert(name, decl)
            .is_none()
        {
            self.declared.push(name);
        }
        self.durations.insert(name, value);
    }

    /// Fold an expression that has to be a duration or a stretch.
    fn duration_of(&self, expr: &Expr, line: usize) -> Result<Timed> {
        let vars = self.param_vars.as_ref();
        match timed_of(expr, line, vars, &self.durations, &self.arrays)? {
            Timed::Number(value) => Err(parse_error(
                line,
                format!("`{expr}` is the number {value} where a duration belongs; give it a unit"),
            )),
            timed => Ok(timed),
        }
    }

    /// Check the length a `delay` or `box` names, which has no further effect.
    fn span_of(&self, expr: &Expr, what: &str, line: usize) -> Result<()> {
        if let Timed::Duration(duration) = self.duration_of(expr, line)?
            && duration.is_negative()
        {
            return Err(parse_error(
                line,
                format!("`{what}` needs a non-negative duration, got `{expr}`"),
            ));
        }
        Ok(())
    }

    fn assign_classical(
        &mut self,
        target: &'a str,
        op: Option<AssignOp>,
        value: &Expr<'a>,
        line: usize,
    ) -> Result<()> {
        if let Some(var) = self.runtime_var(target) {
            return self.assign_runtime(var, op, value, line);
        }
        let Some(decl) = self.classical.get(target) else {
            return Err(parse_error(
                line,
                format!("`{target}` is not a declared classical variable"),
            ));
        };
        let (kind, constant) = (decl.ty, decl.constant);
        if constant {
            return Err(parse_error(
                line,
                format!("`{target}` is `const` and cannot be assigned"),
            ));
        }
        self.reject_guarded_write(target, line)?;
        if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
            return self.assign_duration(target, kind, op, value, line);
        }
        let folded = self.fold_typed(kind, value, line)?;
        let updated = match op {
            None => folded,
            Some(op) => {
                let current = self.typed_of(&Expr::Ident(target), line)?;
                syntax_expr::arithmetic(op.binary(), current, folded, line)?
            }
        };
        self.bind_classical(target, kind, updated, false);
        Ok(())
    }

    fn declare_array(&mut self, decl: &ast::ArrayDecl<'a>, line: usize) -> Result<()> {
        let kind = match decl.ty {
            "int" => ClassicalType::Int(RuntimeType::Int { width: 64 }),
            "uint" => ClassicalType::Int(RuntimeType::Uint { width: 64 }),
            "bool" => ClassicalType::Bool,
            "float" => ClassicalType::FLOAT,
            "angle" => ClassicalType::Angle(None),
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "`array[{other}, ...]`; an array holds `int`, `uint`, `float`, \
                         `angle` or `bool`, and bits belong in a `bit[n]` register"
                    ),
                    line,
                });
            }
        };
        self.reject_redeclaration(decl.name, line)?;
        let mut dims = Vec::with_capacity(decl.dims.len());
        let mut total = 1usize;
        for dim in &decl.dims {
            let size = self.integer_of(dim, line)?;
            let size = usize::try_from(size)
                .ok()
                .filter(|&size| size > 0)
                .ok_or_else(|| parse_error(line, format!("array size must be > 0, got {size}")))?;
            total = total
                .checked_mul(size)
                .filter(|&total| total <= MAX_ARRAY_ELEMENTS)
                .ok_or_else(|| {
                    parse_error(
                        line,
                        format!(
                            "array `{}` holds more than {MAX_ARRAY_ELEMENTS} elements",
                            decl.name
                        ),
                    )
                })?;
            dims.push(size);
        }
        if decl.constant && decl.init.is_none() {
            return Err(parse_error(
                line,
                format!("`const array {}` needs a value", decl.name),
            ));
        }
        let mut values = Vec::with_capacity(total);
        match &decl.init {
            None => values.resize(total, kind.store(ClassicalValue::Int(0))),
            Some(init) => self.fold_initializer(kind, init, &dims, decl.name, line, &mut values)?,
        }
        let array = ClassicalArray {
            ty: kind,
            constant: decl.constant,
            dims,
            values,
        };
        if self.arrays.insert(decl.name, Rc::new(array)).is_none() {
            self.declared.push(decl.name);
        }
        Ok(())
    }

    /// Fold an initializer into `values` in row-major order, checking that each
    /// braced list has one entry per index of the dimension it stands at.
    fn fold_initializer(
        &self,
        kind: ClassicalType,
        init: &ast::ArrayInit,
        dims: &[usize],
        name: &str,
        line: usize,
        values: &mut Vec<ClassicalValue>,
    ) -> Result<()> {
        match (init, dims.split_first()) {
            (ast::ArrayInit::Value(expr), None) => {
                values.push(self.fold_typed(kind, expr, line)?);
                Ok(())
            }
            (ast::ArrayInit::List(entries), Some((&size, inner))) if entries.len() == size => {
                for entry in entries {
                    self.fold_initializer(kind, entry, inner, name, line, values)?;
                }
                Ok(())
            }
            (ast::ArrayInit::List(entries), Some((&size, _))) => Err(parse_error(
                line,
                format!(
                    "initializer for `{name}` lists {} entries where a dimension of size \
                     {size} stands",
                    entries.len()
                ),
            )),
            _ => Err(parse_error(
                line,
                format!("initializer for `{name}` does not match its dimensions"),
            )),
        }
    }

    fn assign_element(&mut self, assign: &ast::ElementAssign<'a>, line: usize) -> Result<()> {
        let Some(array) = self.arrays.get(assign.array) else {
            return Err(parse_error(
                line,
                format!(
                    "`{}` is not a name an assignment can write",
                    Expr::Element(Box::new(syntax_expr::Element {
                        array: assign.array,
                        indices: assign.indices.clone(),
                    }))
                ),
            ));
        };
        if array.constant {
            return Err(parse_error(
                line,
                format!("`{}` is `const` and cannot be assigned", assign.array),
            ));
        }
        self.reject_guarded_write(assign.array, line)?;
        let at = element_offset(array, assign.array, &assign.indices, line, |index| {
            self.typed_of(index, line)
        })?;
        let kind = array.ty;
        let current = array.values[at];
        let folded = self.fold_typed(kind, &assign.value, line)?;
        let updated = kind.store(match assign.op {
            None => folded,
            Some(op) => syntax_expr::arithmetic(op.binary(), current, folded, line)?,
        });
        if let Some(array) = self.arrays.get_mut(assign.array) {
            Rc::make_mut(array).values[at] = updated;
        }
        Ok(())
    }

    fn assign_duration(
        &mut self,
        target: &'a str,
        kind: ClassicalType,
        op: Option<AssignOp>,
        value: &Expr<'a>,
        line: usize,
    ) -> Result<()> {
        if kind == ClassicalType::Stretch {
            return Err(parse_error(
                line,
                format!(
                    "`{target}` is a stretch, which a scheduler sizes and a program cannot assign"
                ),
            ));
        }
        let value = timed_of(
            value,
            line,
            self.param_vars.as_ref(),
            &self.durations,
            &self.arrays,
        )?;
        let updated = match op {
            None => value,
            Some(op) => {
                let current = self
                    .durations
                    .get(target)
                    .copied()
                    .unwrap_or(Timed::Stretch);
                syntax_expr::combine_timed(op.binary(), current, value, line)?
            }
        };
        if let Timed::Number(number) = updated {
            return Err(parse_error(
                line,
                format!("assignment leaves duration `{target}` holding the number {number}"),
            ));
        }
        self.bind_duration(target, kind, updated, false);
        Ok(())
    }

    /// Fold an initializer to the value its declared type holds, naming an
    /// `input` rather than leaving it to read as an unknown identifier. An
    /// integer or `bool` takes a whole number, which a fractional float is not.
    fn fold_typed(&self, kind: ClassicalType, value: &Expr, line: usize) -> Result<ClassicalValue> {
        if let Some(name) = self.input_names.iter().find(|name| value.mentions(name)) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "input `{name}` in a classical declaration; an input binds an angle whole"
                ),
                line,
            });
        }
        Ok(kind.store(match kind {
            ClassicalType::Int(_) | ClassicalType::Bool => {
                ClassicalValue::Int(self.int_of(value, line)?)
            }
            _ => self.typed_of(value, line)?,
        }))
    }

    fn declare_alias(&mut self, name: &'a str, sources: &[Operand<'a>], line: usize) -> Result<()> {
        self.reject_redeclaration(name, line)?;
        let mut kind: Option<ast::RegisterKind> = None;
        let mut indices: Vec<usize> = Vec::new();
        for source in sources {
            let (source_kind, resolved) = self.alias_source(source, line)?;
            match kind {
                Some(seen) if seen != source_kind => {
                    return Err(parse_error(
                        line,
                        format!("alias `{name}` joins qubits and classical bits"),
                    ));
                }
                _ => kind = Some(source_kind),
            }
            indices.extend(resolved);
        }
        let kind =
            kind.ok_or_else(|| parse_error(line, format!("alias `{name}` names nothing")))?;
        Rc::make_mut(&mut self.aliases).insert(name, Alias { kind, indices });
        Ok(())
    }

    /// One operand of an alias, resolved on the side of the register wall the
    /// name it opens with sits on.
    fn alias_source(
        &self,
        source: &Operand,
        line: usize,
    ) -> Result<(ast::RegisterKind, SmallVec<[usize; 4]>)> {
        let Some(name) = source.register() else {
            return Ok((ast::RegisterKind::Qubit, self.qubits_of(source)?));
        };
        let alias_kind = self.alias(name).map(|alias| alias.kind);
        if self.qregs.contains_key(name) || alias_kind == Some(ast::RegisterKind::Qubit) {
            return Ok((ast::RegisterKind::Qubit, self.qubits_of(source)?));
        }
        if self.cregs.contains_key(name) || alias_kind == Some(ast::RegisterKind::Classical) {
            return Ok((ast::RegisterKind::Classical, self.bits_of(source)?));
        }
        Err(PrismError::UndefinedRegister {
            name: name.to_string(),
            line,
        })
    }

    fn declare_def(
        &mut self,
        name: &'a str,
        args: &[DefParam<'a>],
        result: Option<&ast::BitResult<'a>>,
        body: &Block<'a>,
        line: usize,
    ) -> Result<()> {
        let Some((last, rest)) = body.split_last() else {
            if result.is_some() {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares a `bit` result but its body is empty"),
                ));
            }
            Rc::make_mut(&mut self.def_defs).insert(
                name,
                DefDefinition {
                    args: args.to_vec(),
                    result: None,
                    local: None,
                    body: Block::new(),
                },
            );
            return Ok(());
        };
        // A `return` anywhere but last decides at run time whether the rest of
        // the body runs, which no guarded region can say.
        let early = return_line(rest).or_else(|| match last.kind {
            StmtKind::Return(_) => None,
            _ => return_line(std::slice::from_ref(last)),
        });
        if let Some(at) = early {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`return` before the end of def `{name}`, which needs control flow \
                     beyond a guarded region"
                ),
                line: at,
            });
        }
        let local = Self::def_result_local(name, args, result.is_some(), last, rest)?;
        Self::check_def_writes(name, args, local, body, true)?;
        Rc::make_mut(&mut self.def_defs).insert(
            name,
            DefDefinition {
                args: args.to_vec(),
                result: result.cloned(),
                local,
                body: body.clone(),
            },
        );
        Ok(())
    }

    /// The bit a `def` body declares and returns by name, after checking that
    /// its final statement returns what the signature promises.
    fn def_result_local(
        name: &str,
        args: &[DefParam<'a>],
        has_result: bool,
        last: &Stmt<'a>,
        rest: &[Stmt<'a>],
    ) -> Result<Option<&'a str>> {
        let value = match &last.kind {
            StmtKind::Return(value) => value.as_ref(),
            _ if has_result => None,
            _ => return Ok(None),
        };
        let Some(value) = value else {
            if has_result {
                return Err(parse_error(
                    last.line,
                    format!(
                        "def `{name}` declares a `bit` result but does not end by returning one"
                    ),
                ));
            }
            return Ok(None);
        };
        if !has_result {
            return Err(parse_error(
                last.line,
                format!("def `{name}` returns a value but declares no `-> bit` result"),
            ));
        }
        let ast::ReturnValue::Value(expr) = value else {
            return Ok(None);
        };
        let declared = |local: &str| {
            rest.iter().any(|stmt| {
                matches!(
                    &stmt.kind,
                    StmtKind::RegisterDecl {
                        kind: ast::RegisterKind::Classical,
                        name,
                        ..
                    } if *name == local
                )
            })
        };
        match expr.as_ident() {
            Some(local) if is_bit_param(args, local) => Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "def `{name}` returning its bit parameter `{local}`, a copy the \
                     instruction list has no classical move for"
                ),
                line: last.line,
            }),
            Some(local) if declared(local) => Ok(Some(local)),
            _ => Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "return of `{expr}` from def `{name}`; a def returns a measurement or a \
                     bit it declares"
                ),
                line: last.line,
            }),
        }
    }

    /// Check that a `def` body writes classical bits only through its result.
    ///
    /// A bit parameter is a copy, and any other bit would need a classical
    /// register the program never declared, so neither can be written here.
    fn check_def_writes(
        name: &str,
        args: &[DefParam<'a>],
        local: Option<&str>,
        block: &Block<'a>,
        top: bool,
    ) -> Result<()> {
        with_stack(|| {
            for stmt in block {
                match &stmt.kind {
                    StmtKind::RegisterDecl {
                        kind: ast::RegisterKind::Classical,
                        name: declared,
                        ..
                    } if !(top && local == Some(*declared)) => {
                        return Err(PrismError::UnsupportedConstruct {
                            construct: format!(
                                "`bit {declared}` inside def `{name}`, which does not return it; \
                                 holding it would need a classical bit the program never declared"
                            ),
                            line: stmt.line,
                        });
                    }
                    StmtKind::Measure(measure) => {
                        let target = measure.target.register();
                        if target.is_some() && target == local {
                            continue;
                        }
                        let construct = match target {
                            Some(param) if is_bit_param(args, param) => format!(
                                "measure into bit parameter `{param}` of def `{name}`, which is \
                                 passed by value"
                            ),
                            _ => format!(
                                "measure into `{}` inside def `{name}`, which writes classical \
                                 bits only through its result",
                                measure.target.describe()
                            ),
                        };
                        return Err(PrismError::UnsupportedConstruct {
                            construct,
                            line: stmt.line,
                        });
                    }
                    StmtKind::If(conditional) => {
                        Self::check_def_writes(name, args, local, &conditional.then_body, false)?;
                        if let Some(body) = &conditional.else_body {
                            Self::check_def_writes(name, args, local, body, false)?;
                        }
                    }
                    StmtKind::For { body, .. } | StmtKind::Box { body, .. } => {
                        Self::check_def_writes(name, args, local, body, false)?;
                    }
                    StmtKind::Switch { arms, .. } => {
                        for arm in arms {
                            Self::check_def_writes(name, args, local, &arm.body, false)?;
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        })
    }

    // ------------------------------------------------------------- operands

    /// Qubits an operand names: a physical index, a whole register, one
    /// element, a slice, or an alias of any of those.
    pub(super) fn qubits_of(&self, operand: &Operand) -> Result<SmallVec<[usize; 4]>> {
        let name = match &operand.name {
            OperandName::Physical(index) => return Ok(smallvec![*index]),
            OperandName::Register(name) => name,
        };
        if let Some(indices) = self.alias_indices(ast::RegisterKind::Qubit, name, operand)? {
            return Ok(indices);
        }
        self.register_indices(&self.qregs, ast::RegisterKind::Qubit, name, operand)
    }

    pub(super) fn bits_of(&self, operand: &Operand) -> Result<SmallVec<[usize; 4]>> {
        let Some(name) = operand.register() else {
            return Err(parse_error(
                operand.line,
                format!(
                    "`{}` is a qubit where a classical bit belongs",
                    operand.describe()
                ),
            ));
        };
        if let Some(indices) = self.alias_indices(ast::RegisterKind::Classical, name, operand)? {
            return Ok(indices);
        }
        self.register_indices(&self.cregs, ast::RegisterKind::Classical, name, operand)
    }

    pub(super) fn bit_of(&self, operand: &Operand) -> Result<usize> {
        match self.bits_of(operand)?.as_slice() {
            [single] => Ok(*single),
            other => Err(parse_error(
                operand.line,
                format!(
                    "`{}` names {} bits where one belongs",
                    operand.describe(),
                    other.len()
                ),
            )),
        }
    }

    fn alias_indices(
        &self,
        kind: ast::RegisterKind,
        name: &str,
        operand: &Operand,
    ) -> Result<Option<SmallVec<[usize; 4]>>> {
        let Some(alias) = self.alias(name) else {
            return Ok(None);
        };
        if alias.kind != kind {
            return Err(parse_error(
                operand.line,
                format!("alias `{name}` names {}s", alias.kind.name()),
            ));
        }
        let Some(index) = &operand.index else {
            return Ok(Some(alias.indices.iter().copied().collect()));
        };
        let picks = self.pick_indices(alias.indices.len(), index, kind, operand)?;
        Ok(Some(picks.iter().map(|at| alias.indices[*at]).collect()))
    }

    fn register_indices(
        &self,
        registers: &HashMap<&'a str, Register>,
        kind: ast::RegisterKind,
        name: &str,
        operand: &Operand,
    ) -> Result<SmallVec<[usize; 4]>> {
        let register = registers
            .get(name)
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: name.to_string(),
                line: operand.line,
            })?;
        let Some(index) = &operand.index else {
            return Ok((0..register.size).map(|at| register.offset + at).collect());
        };
        let picks = self.pick_indices(register.size, index, kind, operand)?;
        Ok(picks.iter().map(|at| register.offset + at).collect())
    }

    /// Positions a subscript names on a run of `size`.
    fn pick_indices(
        &self,
        size: usize,
        index: &Index,
        kind: ast::RegisterKind,
        operand: &Operand,
    ) -> Result<SmallVec<[usize; 4]>> {
        let line = operand.line;
        let from_end = |value: i64| {
            if value < 0 {
                value + size as i64
            } else {
                value
            }
        };
        let bound = |value: i64| -> Result<usize> {
            let at = usize::try_from(from_end(value)).map_err(|_| {
                parse_error(
                    line,
                    format!(
                        "index {value} in `{}` is before its start",
                        operand.describe()
                    ),
                )
            })?;
            if at >= size {
                return Err(invalid_index(kind, at, size));
            }
            Ok(at)
        };
        match index {
            Index::Single(expr) => Ok(smallvec![bound(self.integer_of(expr, line)?)?]),
            Index::Set(entries) => {
                let mut out = SmallVec::new();
                for entry in entries {
                    out.push(bound(self.integer_of(entry, line)?)?);
                }
                Ok(out)
            }
            Index::Range(range) => {
                let at = |bound: &Option<Expr>, fallback: i64| -> Result<i64> {
                    match bound {
                        None => Ok(fallback),
                        Some(expr) => self.integer_of(expr, line),
                    }
                };
                let first = from_end(at(&range.start, 0)?);
                let last = from_end(at(&range.stop, size as i64 - 1)?);
                let stride = at(&range.step, 1)?;
                if stride == 0 {
                    return Err(parse_error(
                        line,
                        format!("range step in `{}` must be non-zero", operand.describe()),
                    ));
                }
                let mut out = SmallVec::new();
                let mut at = first;
                while (stride > 0 && at <= last) || (stride < 0 && at >= last) {
                    out.push(bound(at)?);
                    // A step past the integer range is past `last` too.
                    let Some(next) = at.checked_add(stride) else {
                        break;
                    };
                    at = next;
                }
                if out.is_empty() {
                    return Err(parse_error(
                        line,
                        format!("`{}` names no index", operand.describe()),
                    ));
                }
                Ok(out)
            }
        }
    }

    // ------------------------------------------------------------ expressions

    pub(super) fn value_of(&self, expr: &Expr, line: usize) -> Result<f64> {
        Ok(self.typed_of(expr, line)?.to_float())
    }

    pub(super) fn typed_of(&self, expr: &Expr, line: usize) -> Result<ClassicalValue> {
        self.reject_runtime(expr, line)?;
        let vars = self.param_vars.as_ref();
        if !self.classical.is_empty() && expr.reads_element() {
            let resolved = replace_elements(expr, &|element| {
                let Some(at) = self.int_bit(element, line)? else {
                    return Ok(None);
                };
                let value = vars.and_then(|vars| vars.get(element.array));
                Ok(value.map(|value| ClassicalValue::Int((value.to_int() >> at) & 1)))
            })?;
            return number_of(&resolved, line, vars, &self.durations, &self.arrays);
        }
        number_of(expr, line, vars, &self.durations, &self.arrays)
    }

    /// The bit `int[n] x` or `uint[n] x` reads as `x[i]`, `None` for any other
    /// element. A negative index counts from the top bit.
    pub(super) fn int_bit(
        &self,
        element: &syntax_expr::Element,
        line: usize,
    ) -> Result<Option<u32>> {
        use crate::circuit::dynamic::ClassicalType as Ty;
        if self.arrays.contains_key(element.array) {
            return Ok(None);
        }
        let Some(ClassicalDecl {
            ty: ClassicalType::Int(Ty::Int { width } | Ty::Uint { width }),
            ..
        }) = self.classical.get(element.array)
        else {
            return Ok(None);
        };
        let name = element.array;
        let [index] = element.indices.as_slice() else {
            return Err(parse_error(
                line,
                format!("`{name}` is an integer and takes one bit index"),
            ));
        };
        let written = self.int_of(index, line)?;
        let at = if written < 0 {
            written + i128::from(*width)
        } else {
            written
        };
        if !(0..i128::from(*width)).contains(&at) {
            return Err(parse_error(
                line,
                format!("bit {written} is outside `{name}`, which holds {width} bits"),
            ));
        }
        Ok(Some(at as u32))
    }

    /// Fold `expr` to an integer: an integer exactly, a float only when whole.
    fn int_of(&self, expr: &Expr, line: usize) -> Result<i128> {
        match self.typed_of(expr, line)? {
            ClassicalValue::Float(value) if !value.is_finite() || value.fract() != 0.0 => Err(
                parse_error(line, format!("expected an integer, got {value}")),
            ),
            value => Ok(value.to_int()),
        }
    }

    pub(super) fn integer_of(&self, expr: &Expr, line: usize) -> Result<i64> {
        i64::try_from(self.int_of(expr, line)?)
            .map_err(|_| parse_error(line, "integer expression out of range"))
    }

    // ------------------------------------------------------------------ calls

    fn exec_call(
        &mut self,
        modifiers: &[ast::Modifier],
        name: &str,
        params: &[Argument],
        operands: &[Operand],
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        if self.dynamic.is_some() && params.iter().any(|param| self.argument_is_runtime(param)) {
            return self.exec_runtime_rotation(modifiers, name, params, operands, line, out);
        }
        let modifiers = self.fold_modifiers(modifiers, line)?;

        if self.def_defs.contains_key(name) {
            let instrs = self.expand_def(name, params, None, line)?;
            out.extend(Self::modify_expansion(instrs, &modifiers, name, line)?);
            return Ok(());
        }

        let (values, input_slot) = self.fold_params(params, line)?;
        if name == "gphase" {
            let qubits = operands
                .iter()
                .map(|operand| match self.qubits_of(operand)?.as_slice() {
                    [single] => Ok(*single),
                    _ => Err(parse_error(
                        line,
                        format!(
                            "`{}` names a register where one qubit belongs",
                            operand.describe()
                        ),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            out.extend(self.resolve_global_phase(&values, &modifiers, &qubits, line)?);
            return Ok(());
        }
        if input_slot.is_some() && !modifiers.is_empty() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("modifier on `{name}` reading an `input`"),
                line,
            });
        }

        let resolved: SmallVec<[SmallVec<[usize; 4]>; 4]> = operands
            .iter()
            .map(|operand| self.qubits_of(operand))
            .collect::<Result<_>>()?;
        let width = self.broadcast_length(&resolved, name, line)?;

        let base = out.len();
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
            self.resolve_gate_application_once(
                name,
                &values,
                &modifiers,
                &qubits,
                input_slot.is_some(),
                line,
                out,
            )?;
        }
        if input_slot.is_some() {
            Self::check_every_instruction_is_bindable(&out[base..], name, line)?;
        }
        self.pending_input_slot = input_slot;
        Ok(())
    }

    fn fold_modifiers(
        &self,
        modifiers: &[ast::Modifier],
        line: usize,
    ) -> Result<Vec<super::Modifier>> {
        let mut out = Vec::with_capacity(modifiers.len());
        let mut controls = 0usize;
        for modifier in modifiers {
            match modifier {
                ast::Modifier::Inv => out.push(super::Modifier::Inv),
                ast::Modifier::Ctrl { negated, count } => {
                    let count = match count {
                        None => 1,
                        Some(expr) => self.control_count(expr, line)?,
                    };
                    // Checked over the whole chain before any entry is built, so
                    // a long run of `ctrl(255) @` costs nothing to refuse.
                    controls += count;
                    if controls > usize::from(u8::MAX) {
                        return Err(PrismError::UnsupportedConstruct {
                            construct: format!("ctrl @ chain past {} controls", u8::MAX),
                            line,
                        });
                    }
                    out.extend((0..count).map(|_| super::Modifier::Ctrl { negated: *negated }));
                }
                ast::Modifier::Pow(expr) => {
                    let exponent = self.value_of(expr, line)?;
                    if exponent.fract() == 0.0 && exponent.abs() > MAX_POW_REPEATS as f64 {
                        return Err(parse_error(
                            line,
                            format!(
                                "pow({exponent}) repeats a gate more than \
                                 {MAX_POW_REPEATS} times"
                            ),
                        ));
                    }
                    out.push(super::Modifier::Pow(exponent));
                }
            }
        }
        Ok(out)
    }

    /// The `n` of `ctrl(n) @`: a positive integer, at most the widest control
    /// chain a gate can carry.
    fn control_count(&self, expr: &Expr, line: usize) -> Result<usize> {
        let count = self.integer_of(expr, line)?;
        if count < 1 {
            return Err(parse_error(
                line,
                format!("`ctrl({expr})` needs a positive control count, got {count}"),
            ));
        }
        if count > i64::from(u8::MAX) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("ctrl @ chain past {} controls", u8::MAX),
                line,
            });
        }
        Ok(count as usize)
    }

    /// Initialize a fresh `bit` register from a bit string, written most
    /// significant first, or a value. The register starts cleared, so only an
    /// initializer that sets a bit writes one, which only a dynamic program can.
    fn init_bits(
        &mut self,
        name: &str,
        size: Option<&Expr<'a>>,
        init: &ast::BitInit<'a>,
        line: usize,
    ) -> Result<()> {
        let width = match size {
            Some(size) => self.integer_of(size, line)?,
            None => 1,
        };
        let bits = self.bits_of(&Operand {
            name: OperandName::Register(name),
            index: None,
            line,
        })?;
        let values: Vec<bool> = match init {
            ast::BitInit::Bits(text) => {
                let digits: Vec<char> = text.chars().filter(|c| *c != '_').collect();
                if let Some(bad) = digits.iter().find(|c| !matches!(c, '0' | '1')) {
                    return Err(parse_error(
                        line,
                        format!("`\"{text}\"` is not a bit string: `{bad}` is neither 0 nor 1"),
                    ));
                }
                if digits.len() as i64 != width {
                    return Err(parse_error(
                        line,
                        format!(
                            "`\"{text}\"` has {} bit(s) but `{name}` declares {width}",
                            digits.len()
                        ),
                    ));
                }
                digits.iter().rev().map(|digit| *digit == '1').collect()
            }
            ast::BitInit::Value(expr) if self.reads_runtime(expr) => {
                let value = self.lower_expr(expr, line)?;
                return self.write_classical_bits(&bits, value, line);
            }
            ast::BitInit::Value(expr) => {
                let word = self.int_of(expr, line)?;
                (0..bits.len())
                    .map(|at| {
                        if at < 128 {
                            (word >> at) & 1 == 1
                        } else {
                            word < 0
                        }
                    })
                    .collect()
            }
        };
        if !values.contains(&true) {
            return Ok(());
        }
        if self.dynamic.is_none() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`{name}` initialized with a set bit; the instruction list has no classical \
                     write, and `openqasm::parse_dynamic` carries one"
                ),
                line,
            });
        }
        self.write_constant_bits(&bits, &values, line)
    }

    /// Evaluate a call's angle arguments, reporting the one `input` slot they
    /// read. Two inputs on one gate are rejected: a slot is written onto the
    /// single angle a bindable gate carries.
    fn fold_params(&self, params: &[Argument], line: usize) -> Result<(Vec<f64>, Option<usize>)> {
        let mut values = Vec::with_capacity(params.len());
        let mut input_slot = None;
        for param in params {
            let element = self.element_argument(param);
            let expr = match (param, &element) {
                (_, Some(element)) => element,
                (Argument::Value(expr), None) => expr,
                (Argument::Operand(_), None) => {
                    return Err(parse_error(
                        line,
                        "a gate angle cannot be a qubit reference",
                    ));
                }
            };
            let (value, slot) = self.fold_angle(expr, line)?;
            if slot.is_some() && input_slot.is_some() {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "two `input` parameters on one gate".to_string(),
                    line,
                });
            }
            input_slot = input_slot.or(slot);
            values.push(value);
        }
        Ok((values, input_slot))
    }

    /// An argument written `a[i]` parses as a qubit or bit reference; read it
    /// as an array element when `a` names an array.
    fn element_argument<'e>(&self, argument: &Argument<'e>) -> Option<Expr<'e>> {
        let Argument::Operand(operand) = argument else {
            return None;
        };
        let array = operand.register().filter(|name| {
            self.arrays.contains_key(name)
                || matches!(
                    self.classical.get(name),
                    Some(ClassicalDecl {
                        ty: ClassicalType::Int(_),
                        ..
                    })
                )
        })?;
        let Some(Index::Single(index)) = &operand.index else {
            return None;
        };
        Some(Expr::Element(Box::new(syntax_expr::Element {
            array,
            indices: vec![index.clone()],
        })))
    }

    /// One angle argument, reporting the slot when it is exactly an `input`.
    fn fold_angle(&self, expr: &Expr, line: usize) -> Result<(f64, Option<usize>)> {
        if let Some((name, &slot)) = expr
            .as_ident()
            .and_then(|name| self.inputs.get(name).map(|slot| (name, slot)))
        {
            if self.nested {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("input `{name}` inside a block body"),
                    line,
                });
            }
            return Ok((0.0, Some(slot)));
        }
        if let Some(name) = self.input_names.iter().find(|name| expr.mentions(name)) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "an expression over input `{name}`; an input binds an angle whole"
                ),
                line,
            });
        }
        Ok((self.value_of(expr, line)?, None))
    }

    /// Expand a user `gate` by binding its qubit names to the call's qubits and
    /// its parameters to the call's values, then running the body.
    pub(super) fn expand_user_gate(
        &self,
        name: &str,
        call_params: &[f64],
        call_qubits: &SmallVec<[usize; 4]>,
        line: usize,
    ) -> Result<Option<Vec<Instruction>>> {
        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(parse_error(
                line,
                format!(
                    "gate expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive gate definition for `{name}`"
                ),
            ));
        }
        let Some(def) = self.gate_defs.get(name) else {
            return Ok(None);
        };
        if call_params.len() != def.params.len() {
            return Err(parse_error(
                line,
                format!(
                    "gate `{name}` expects {} parameters, got {}",
                    def.params.len(),
                    call_params.len()
                ),
            ));
        }
        if call_qubits.len() != def.qubits.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.qubits.len(),
                got: call_qubits.len(),
            });
        }

        // A gate body sees its own parameters and nothing else, which is what
        // the language scopes it to.
        let bindings: HashMap<&'a str, ClassicalValue> = def
            .params
            .iter()
            .copied()
            .zip(
                call_params
                    .iter()
                    .map(|&value| ClassicalValue::Float(value)),
            )
            .collect();
        let mut sub = self.expansion_parser(bindings);
        if let Some(&first) = call_qubits.first() {
            sub.phase_qubit = first;
        }
        for (&qubit_name, qubit) in def.qubits.iter().zip(call_qubits) {
            sub.params.insert(
                qubit_name,
                Alias {
                    kind: ast::RegisterKind::Qubit,
                    indices: vec![*qubit],
                },
            );
        }
        Ok(Some(sub.execute(&def.body)?))
    }

    /// `target = f(...);`: a `def` call whose `bit` result lands on `target`,
    /// or a builtin on the right of a classical assignment.
    fn exec_call_assign(
        &mut self,
        assign: &ast::CallAssign<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        if self.def_defs.contains_key(assign.name) {
            let bits = self.bits_of(&assign.target)?;
            out.extend(self.expand_def(assign.name, &assign.args, Some(&bits), line)?);
            return Ok(());
        }
        let Some(target) = assign
            .target
            .register()
            .filter(|_| assign.target.index.is_none())
        else {
            return Err(parse_error(
                line,
                format!(
                    "`{}` is not a name an assignment can write",
                    assign.target.describe()
                ),
            ));
        };
        let args = assign
            .args
            .iter()
            .map(|arg| match arg {
                Argument::Value(expr) => Ok(expr.clone()),
                Argument::Operand(operand) => Err(parse_error(
                    line,
                    format!(
                        "`{}` is a qubit or bit where a value belongs",
                        operand.describe()
                    ),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let value = Expr::Call(Box::new(syntax_expr::Call {
            name: assign.name,
            args,
        }));
        self.assign_classical(target, None, &value, line)
    }

    /// Inline a `def` call, binding each argument by the kind its declaration
    /// gave it and the result, when there is one, to `result`.
    fn expand_def(
        &self,
        name: &str,
        args: &[Argument],
        result: Option<&[usize]>,
        line: usize,
    ) -> Result<Vec<Instruction>> {
        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(parse_error(
                line,
                format!(
                    "def expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive call to `{name}`"
                ),
            ));
        }
        let def = self.def_defs.get(name).expect("checked by the caller");
        if args.len() != def.args.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.args.len(),
                got: args.len(),
            });
        }

        // The body takes its own copy of the caller's values and constants, and a
        // bit parameter can unshare the register and alias tables, so the copies
        // come out of the expansion budget like the work they stand for.
        let copied = self.param_vars.as_ref().map_or(0, HashMap::len)
            + self.durations.len()
            + self.classical.len()
            + self.arrays.len()
            + self.aliases.len()
            + self.cregs.len();
        self.charge(copied, line)?;
        let mut bindings: HashMap<&'a str, ClassicalValue> =
            self.param_vars.clone().unwrap_or_default();
        let mut durations = self.durations.clone();
        let mut qubit_bindings: Vec<(&'a str, usize)> = Vec::new();
        let mut bit_bindings: Vec<(&'a str, SmallVec<[usize; 4]>)> = Vec::new();
        for (slot, argument) in def.args.iter().zip(args) {
            match slot {
                DefParam::Qubit(param_name) => {
                    let param_name = *param_name;
                    let operand = match argument {
                        Argument::Operand(operand) => operand.clone(),
                        Argument::Value(expr) => {
                            let Some(register) = expr.as_ident() else {
                                return Err(parse_error(
                                    line,
                                    format!(
                                        "def `{name}` qubit parameter `{param_name}` needs a \
                                         qubit, not an expression"
                                    ),
                                ));
                            };
                            Operand {
                                name: OperandName::Register(register),
                                index: None,
                                line,
                            }
                        }
                    };
                    match self.qubits_of(&operand)?.as_slice() {
                        [single] => qubit_bindings.push((param_name, *single)),
                        _ => {
                            return Err(parse_error(
                                line,
                                format!(
                                    "def `{name}` qubit parameter `{param_name}` requires a \
                                     single qubit, got register `{}`",
                                    operand.describe()
                                ),
                            ));
                        }
                    }
                }
                DefParam::Value {
                    name: param_name,
                    integral,
                } => {
                    let param_name = *param_name;
                    let element = self.element_argument(argument);
                    let expr = match (argument, &element) {
                        (_, Some(element)) => element,
                        (Argument::Value(expr), None) => expr,
                        (Argument::Operand(_), None) => {
                            return Err(parse_error(
                                line,
                                format!("def `{name}` parameter `{param_name}` takes a value"),
                            ));
                        }
                    };
                    // The body reads the argument's value, so an input bound
                    // later would never reach it.
                    if let Some(input) = self.input_names.iter().find(|input| expr.mentions(input))
                    {
                        return Err(PrismError::UnsupportedConstruct {
                            construct: format!("input `{input}` as an argument to `def {name}`"),
                            line,
                        });
                    }
                    let value = number_of(expr, line, Some(&bindings), &durations, &self.arrays)?;
                    let value = match value {
                        ClassicalValue::Float(float) if *integral && float.fract() != 0.0 => {
                            return Err(parse_error(
                                line,
                                format!("def `{name}` parameter `{param_name}` takes an integer"),
                            ));
                        }
                        value if *integral => ClassicalValue::Int(value.to_int()),
                        value => value,
                    };
                    bindings.insert(param_name, value);
                }
                DefParam::Duration(param_name) => {
                    let Argument::Value(expr) = argument else {
                        return Err(parse_error(
                            line,
                            format!("def `{name}` parameter `{param_name}` takes a duration"),
                        ));
                    };
                    let value = timed_of(expr, line, Some(&bindings), &durations, &self.arrays)?;
                    if let Timed::Number(number) = value {
                        return Err(parse_error(
                            line,
                            format!(
                                "def `{name}` parameter `{param_name}` takes a duration, got {number}"
                            ),
                        ));
                    }
                    durations.insert(param_name, value);
                }
                DefParam::Bit {
                    name: param_name,
                    width,
                } => {
                    let operand = match argument {
                        Argument::Operand(operand) => operand.clone(),
                        Argument::Value(expr) => match expr.as_ident() {
                            Some(register) => Operand {
                                name: OperandName::Register(register),
                                index: None,
                                line,
                            },
                            None => {
                                return Err(parse_error(
                                    line,
                                    format!(
                                        "def `{name}` bit parameter `{param_name}` needs a bit, \
                                         not an expression"
                                    ),
                                ));
                            }
                        },
                    };
                    let bits = self.bits_of(&operand)?;
                    let wanted = match width {
                        Some(width) => self.integer_of(width, line)?,
                        None => 1,
                    };
                    if bits.len() as i64 != wanted {
                        return Err(parse_error(
                            line,
                            format!(
                                "def `{name}` bit parameter `{param_name}` takes {wanted} bit(s), \
                                 got {}",
                                bits.len()
                            ),
                        ));
                    }
                    bit_bindings.push((param_name, bits));
                }
            }
        }

        let frame = match (&def.result, result) {
            (None, None) => DefResult {
                bits: SmallVec::new(),
                local: None,
            },
            (None, Some(_)) => {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares no result to assign"),
                ));
            }
            (Some(_), None) => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "a call to def `{name}` that drops its `bit` result, which would need \
                         a classical bit the program never declared"
                    ),
                    line,
                });
            }
            (Some(declared), Some(bits)) => {
                let wanted = match &declared.width {
                    Some(width) => self.integer_of(width, line)?,
                    None => 1,
                };
                if bits.len() as i64 != wanted {
                    return Err(parse_error(
                        line,
                        format!(
                            "def `{name}` returns {wanted} bit(s) where its target names {}",
                            bits.len()
                        ),
                    ));
                }
                // The named result is written in place, so a parameter reading
                // the same bit would see the new value rather than its copy.
                let aliased = def.local.is_some()
                    && bit_bindings
                        .iter()
                        .any(|(_, param)| param.iter().any(|bit| bits.contains(bit)));
                if aliased {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "a call to def `{name}` passing a bit it also assigns, which the \
                             body would read after writing its result"
                        ),
                        line,
                    });
                }
                DefResult {
                    bits: bits.iter().copied().collect(),
                    local: def.local,
                }
            }
        };

        let mut sub = self.expansion_parser(bindings);
        // The body runs in a parser of its own, so a write it made to an outer
        // variable would be dropped on return; only the constants it may read
        // stay declared, and a write to anything else is rejected.
        sub.classical = Rc::new(
            self.classical
                .iter()
                .filter(|(_, decl)| decl.constant)
                .map(|(name, decl)| (*name, decl.clone()))
                .collect(),
        );
        sub.durations = durations;
        // A def sees the global constants and nothing else it does not take as
        // an argument, so only a `const` array reaches its body.
        sub.arrays = self
            .arrays
            .iter()
            .filter(|(_, array)| array.constant)
            .map(|(name, array)| (*name, Rc::clone(array)))
            .collect();
        for (param_name, qubit) in qubit_bindings {
            sub.params.insert(
                param_name,
                Alias {
                    kind: ast::RegisterKind::Qubit,
                    indices: vec![qubit],
                },
            );
        }
        for (param_name, bits) in bit_bindings {
            sub.bind_bits(param_name, bits);
        }
        if let Some(local) = frame.local {
            sub.bind_bits(local, frame.bits.clone());
        }
        sub.def_result = Some(frame);
        sub.execute(&def.body)
    }

    /// Name `bits` inside an expanded body: as a register when they run
    /// contiguously, which a register comparison needs, and as an alias
    /// otherwise.
    fn bind_bits(&mut self, name: &'a str, bits: SmallVec<[usize; 4]>) {
        let contiguous = bits.windows(2).all(|pair| pair[1] == pair[0] + 1);
        match bits.first() {
            Some(&offset) if contiguous => {
                self.params.remove(name);
                if self.aliases.contains_key(name) {
                    Rc::make_mut(&mut self.aliases).remove(name);
                }
                Rc::make_mut(&mut self.cregs).insert(
                    name,
                    Register {
                        offset,
                        size: bits.len(),
                    },
                );
            }
            _ => {
                self.params.insert(
                    name,
                    Alias {
                        kind: ast::RegisterKind::Classical,
                        indices: bits.to_vec(),
                    },
                );
            }
        }
    }

    /// A parser for an expanded body: the enclosing registers and definitions,
    /// one level deeper, with its own bindings.
    fn expansion_parser(&self, bindings: HashMap<&'a str, ClassicalValue>) -> Parser<'a> {
        Parser {
            dialect: self.dialect,
            input: "",
            qregs: Rc::clone(&self.qregs),
            cregs: Rc::clone(&self.cregs),
            gate_defs: Rc::clone(&self.gate_defs),
            def_defs: Rc::clone(&self.def_defs),
            total_qubits: self.total_qubits,
            total_cbits: self.total_cbits,
            gate_expansion_depth: self.gate_expansion_depth + 1,
            region_depth: self.region_depth,
            exec_depth: self.exec_depth,
            param_vars: Some(bindings),
            inputs: HashMap::new(),
            input_names: Vec::new(),
            links: Vec::new(),
            pending_input_slot: None,
            nested: true,
            results: Vec::new(),
            noise_specs: Vec::new(),
            pending_noise: None,
            verbatim_pending: false,
            physical: self.physical,
            aliases: Rc::clone(&self.aliases),
            params: self.params.clone(),
            classical: Rc::clone(&self.classical),
            durations: HashMap::new(),
            arrays: HashMap::new(),
            declared: Vec::new(),
            def_result: None,
            guard: None,
            dynamic: None,
            expansion_left: Rc::clone(&self.expansion_left),
            phase_qubit: self.phase_qubit,
        }
    }

    // ------------------------------------------------------------ control flow

    fn exec_if(
        &mut self,
        conditional: &ast::Conditional<'a>,
        line: usize,
    ) -> Result<Vec<Instruction>> {
        let general = match &conditional.condition {
            Condition::Expr(_) => true,
            Condition::Truthy(operand)
            | Condition::Negated(operand)
            | Condition::Compare { lhs: operand, .. } => {
                self.single_bit(operand).is_none()
                    && operand.register().is_some_and(|name| {
                        !self.is_bit_name(name)
                            && (self.classical.contains_key(name)
                                || self.arrays.contains_key(name)
                                || syntax_expr::eval(&Expr::Ident(name), line, None).is_ok()
                                || self
                                    .param_vars
                                    .as_ref()
                                    .is_some_and(|vars| vars.contains_key(name)))
                    })
            }
            Condition::Parity { .. } => false,
        };
        if general {
            let expr = super::dynamic::condition_expr(&conditional.condition)?;
            if let ClassicalExpr::Const(value) = self.lower_expr(&expr, line)? {
                let body = if value.truthy() {
                    Some(&conditional.then_body)
                } else {
                    conditional.else_body.as_ref()
                };
                return body.map_or(Ok(Vec::new()), |body| self.exec_box(body));
            }
        }
        let condition = self.condition_of(&conditional.condition, line)?;
        let then_instrs = self.region(&conditional.then_body)?;
        let Some(else_body) = &conditional.else_body else {
            return Ok(guarded(condition, then_instrs).into_iter().collect());
        };
        // The negated region re-reads the classical bits after the `then` body
        // has run, so a body that measures into its own guard bits could take
        // both arms.
        if crate::circuit::body_writes_condition_bits(&then_instrs, &condition) {
            return Err(parse_error(
                line,
                "`else` needs a condition the `if` body does not overwrite; \
                          this body measures into a bit the condition reads",
            ));
        }
        let else_instrs = self.region(else_body)?;
        let mut out = Vec::new();
        out.extend(guarded(condition.clone(), then_instrs));
        out.extend(guarded(condition.negate(), else_instrs));
        Ok(out)
    }

    /// Run a `box` body in place, with no guard and no barrier at either edge.
    ///
    /// A box only fixes timing, which an ideal simulation does not model, so
    /// fusion may merge across its boundary as freely as across a `delay`. The
    /// names its body declares go out of scope with it.
    pub(super) fn exec_box(&mut self, body: &Block<'a>) -> Result<Vec<Instruction>> {
        let scope = self.open_scope();
        let was_nested = std::mem::replace(&mut self.nested, true);
        let result = self.execute(body);
        self.nested = was_nested;
        self.close_scope(scope);
        result
    }

    /// The point to hand [`Parser::close_scope`] when the scope ends.
    pub(super) fn open_scope(&self) -> usize {
        self.declared.len()
    }

    /// Drop the names declared since `scope` was opened. Names declared before
    /// it keep whatever the body wrote to them.
    pub(super) fn close_scope(&mut self, scope: usize) {
        for name in self.declared.split_off(scope) {
            self.forget_runtime(name);
            Rc::make_mut(&mut self.classical).remove(name);
            self.durations.remove(name);
            self.arrays.remove(name);
            if let Some(values) = self.param_vars.as_mut() {
                values.remove(name);
            }
        }
    }

    /// Run a guarded block one nesting level down, in a scope of its own.
    ///
    /// Whether the block runs is decided by a measurement, while a classical
    /// variable is folded here at parse time, so the block may not write any
    /// variable declared outside it.
    pub(super) fn region(&mut self, block: &Block<'a>) -> Result<Vec<Instruction>> {
        self.enter_region_depth(block)?;
        let scope = self.open_scope();
        let outer_guard = self.guard.replace(scope);
        let was_nested = std::mem::replace(&mut self.nested, true);
        self.enter_collecting();
        let body = self.execute(block);
        self.leave_collecting();
        self.nested = was_nested;
        self.guard = outer_guard;
        self.close_scope(scope);
        self.region_depth -= 1;
        body
    }

    fn reject_guarded_write(&self, name: &str, line: usize) -> Result<()> {
        let Some(guard) = self.guard else {
            return Ok(());
        };
        let outer = self.classical.contains_key(name) || self.arrays.contains_key(name);
        if outer && !self.declared[guard..].contains(&name) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "a write to `{name}` under a runtime `if` or `switch`; the variable is \
                     folded at parse time and cannot depend on a measurement, so parse the \
                     program with `openqasm::parse_dynamic` to evaluate it at runtime"
                ),
                line,
            });
        }
        Ok(())
    }

    fn enter_region_depth(&mut self, block: &Block<'a>) -> Result<()> {
        if self.region_depth >= MAX_REGION_DEPTH {
            return Err(parse_error(
                block.first().map_or(1, |stmt| stmt.line),
                format!("`if` blocks nest deeper than {MAX_REGION_DEPTH}"),
            ));
        }
        self.region_depth += 1;
        Ok(())
    }

    fn exec_for(
        &mut self,
        variable: &'a str,
        range: &ForRange<'a>,
        body: &Block<'a>,
        line: usize,
    ) -> Result<Vec<Instruction>> {
        let values = self.for_values(range, line)?;
        if values.len() as i64 > MAX_FOR_ITERATIONS {
            return Err(parse_error(
                line,
                format!(
                    "for loop iterates {} times (max {MAX_FOR_ITERATIONS})",
                    values.len()
                ),
            ));
        }
        let mut out = Vec::new();
        // The loop variable may shadow an outer name, whose value comes back
        // once the loop ends.
        let shadowed = self
            .param_vars
            .as_ref()
            .and_then(|vars| vars.get(variable).copied());
        let scope = self.open_scope();
        let mut result = Ok(());
        self.enter_static_loop();
        for value in values {
            if let Err(error) = self.charge(1, line) {
                result = Err(error);
                break;
            }
            // Each pass drops what the body declared, so a declaration binds
            // for one pass, while a write to an outer name carries on.
            self.param_vars
                .get_or_insert_with(HashMap::new)
                .insert(variable, ClassicalValue::Int(i128::from(value)));
            let was_nested = std::mem::replace(&mut self.nested, true);
            let produced = self.execute(body);
            self.nested = was_nested;
            self.close_scope(scope);
            match produced {
                Ok(produced) => out.extend(produced),
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.leave_static_loop();
        if let Some(vars) = self.param_vars.as_mut() {
            match shadowed {
                Some(value) => vars.insert(variable, value),
                None => vars.remove(variable),
            };
        }
        result.map(|()| out)
    }

    fn for_values(&self, range: &ForRange, line: usize) -> Result<Vec<i64>> {
        match range {
            ForRange::Set(entries) => entries
                .iter()
                .map(|entry| self.integer_of(entry, line))
                .collect(),
            ForRange::Range { start, step, stop } => {
                let first = self.integer_of(start, line)?;
                let last = self.integer_of(stop, line)?;
                let stride = match step {
                    None => 1,
                    Some(expr) => self.integer_of(expr, line)?,
                };
                if stride == 0 {
                    return Err(parse_error(line, "for loop range step must be non-zero"));
                }
                let mut values = Vec::new();
                let mut at = first;
                while (stride > 0 && at <= last) || (stride < 0 && at >= last) {
                    values.push(at);
                    if values.len() as i64 > MAX_FOR_ITERATIONS {
                        return Err(parse_error(
                            line,
                            format!("for loop iterates more than {MAX_FOR_ITERATIONS} times"),
                        ));
                    }
                    let Some(next) = at.checked_add(stride) else {
                        break;
                    };
                    at = next;
                }
                Ok(values)
            }
        }
    }

    fn exec_switch(
        &mut self,
        target: &Expr<'a>,
        arms: &[ast::SwitchArm<'a>],
        line: usize,
    ) -> Result<Vec<Instruction>> {
        let lowered = self.lower_expr(target, line)?;
        if let ClassicalExpr::Const(value) = lowered {
            return self.exec_switch_constant(value, arms, line);
        }
        let (offset, size) =
            register_value(&lowered).ok_or_else(|| PrismError::UnsupportedConstruct {
                construct: format!(
                    "a `switch` on `{target}`, which no guard on classical bits spells; \
                     `openqasm::parse_dynamic` evaluates it at runtime"
                ),
                line,
            })?;
        let mut labels: Vec<u64> = Vec::new();
        let mut cases: Vec<(Vec<u64>, Vec<Instruction>)> = Vec::new();
        let mut default: Option<Vec<Instruction>> = None;
        for arm in arms {
            let body = self.region(&arm.body)?;
            let Some(arm_labels) = &arm.labels else {
                if default.replace(body).is_some() {
                    return Err(parse_error(
                        arm.line,
                        "`switch` has more than one `default` arm",
                    ));
                }
                continue;
            };
            let mut values = Vec::new();
            for label in arm_labels {
                let value = self.integer_of(label, arm.line)?;
                let value = u64::try_from(value).map_err(|_| {
                    parse_error(
                        arm.line,
                        format!("`switch` case label must be non-negative, got `{value}`"),
                    )
                })?;
                if labels.contains(&value) {
                    return Err(parse_error(
                        arm.line,
                        format!("`switch` case label {value} appears twice"),
                    ));
                }
                labels.push(value);
                values.push(value);
            }
            cases.push((values, body));
        }

        // The chain is exclusive only because no arm writes the switched
        // register, which is checked rather than assumed.
        let probe = ClassicalCondition::RegisterEquals {
            offset,
            size,
            value: 0,
        };
        let writes_operand = cases
            .iter()
            .map(|(_, body)| body)
            .chain(default.iter())
            .any(|body| crate::circuit::body_writes_condition_bits(body, &probe));
        if writes_operand {
            return Err(parse_error(
                line,
                "`switch` needs an operand no arm overwrites; \
                          an arm here measures into the switched register",
            ));
        }

        let mut out = Vec::new();
        for (values, body) in &cases {
            for &value in values {
                let condition = ClassicalCondition::RegisterEquals {
                    offset,
                    size,
                    value,
                };
                out.extend(guarded(condition, body.clone()));
            }
        }
        if let Some(body) = default {
            out.extend(nest_default_arm(
                offset,
                size,
                &labels,
                body,
                self.region_depth,
                line,
            )?);
        }
        Ok(out)
    }

    /// The one arm a `switch` on a parse-time value runs, or its `default`.
    fn exec_switch_constant(
        &mut self,
        value: ClassicalValue,
        arms: &[ast::SwitchArm<'a>],
        line: usize,
    ) -> Result<Vec<Instruction>> {
        if matches!(value, ClassicalValue::Float(float) if float.fract() != 0.0) {
            return Err(parse_error(
                line,
                format!("`switch` needs an integer, got {}", value.to_float()),
            ));
        }
        let value = value.to_int();
        let mut labels: Vec<i128> = Vec::new();
        let mut taken = None;
        let mut default = None;
        for arm in arms {
            let Some(arm_labels) = &arm.labels else {
                if default.replace(&arm.body).is_some() {
                    return Err(parse_error(
                        arm.line,
                        "`switch` has more than one `default` arm",
                    ));
                }
                continue;
            };
            for label in arm_labels {
                let label = self.int_of(label, arm.line)?;
                if labels.contains(&label) {
                    return Err(parse_error(
                        arm.line,
                        format!("`switch` case label {label} appears twice"),
                    ));
                }
                labels.push(label);
                if label == value {
                    taken = Some(&arm.body);
                }
            }
        }
        taken
            .or(default)
            .map_or(Ok(Vec::new()), |body| self.exec_box(body))
    }

    /// The bit an unsubscripted name stands for when it names exactly one,
    /// as `bit b;` or a `def`'s `bit` parameter does.
    fn single_bit(&self, operand: &Operand) -> Option<usize> {
        if operand.index.is_some() {
            return None;
        }
        match self.bits_of(operand).ok()?.as_slice() {
            [bit] => Some(*bit),
            _ => None,
        }
    }

    pub(super) fn condition_of(
        &self,
        condition: &Condition,
        line: usize,
    ) -> Result<ClassicalCondition> {
        match condition {
            Condition::Truthy(operand) => {
                if let Some(bit) = self.single_bit(operand) {
                    return Ok(ClassicalCondition::BitIsOne(bit));
                }
                if operand.index.is_none() {
                    return Err(parse_error(
                        line,
                        format!(
                            "expected `{0}==value`, `{0}!=value`, `{0}[i]`, `!{0}[i]` or \
                             `{0}[i]==0/1` in `if` condition, got `{0}`",
                            operand.describe()
                        ),
                    ));
                }
                Ok(ClassicalCondition::BitIsOne(self.bit_of(operand)?))
            }
            Condition::Negated(operand) => {
                if let Some(bit) = self.single_bit(operand) {
                    return Ok(ClassicalCondition::BitIsZero(bit));
                }
                if operand.index.is_none() {
                    return Err(parse_error(
                        line,
                        format!(
                            "expected `!c[i]` form in `if` condition, got `!{}`",
                            operand.describe()
                        ),
                    ));
                }
                Ok(ClassicalCondition::BitIsZero(self.bit_of(operand)?))
            }
            Condition::Parity { bits, compare } => {
                let expected = match compare {
                    None => true,
                    Some((op, rhs)) => {
                        let negate = *op == CmpOp::NotEqual;
                        match self.integer_of(rhs, line)? {
                            0 => negate,
                            1 => !negate,
                            other => {
                                return Err(parse_error(
                                    line,
                                    format!("a parity compares against 0 or 1, got `{other}`"),
                                ));
                            }
                        }
                    }
                };
                let mut resolved: Vec<usize> = Vec::with_capacity(bits.len());
                for bit in bits {
                    if bit.index.is_none() {
                        return Err(parse_error(
                            line,
                            format!("`{}` is a register where a bit belongs", bit.describe()),
                        ));
                    }
                    resolved.push(self.bit_of(bit)?);
                }
                Ok(ClassicalCondition::Parity {
                    bits: resolved.into_boxed_slice(),
                    expected,
                })
            }
            Condition::Expr(expr) => {
                guard_condition(&self.lower_expr(expr, line)?).ok_or_else(|| {
                    PrismError::UnsupportedConstruct {
                        construct: format!(
                            "condition `{expr}`, which no guard on classical bits spells; \
                             `openqasm::parse_dynamic` evaluates it at runtime"
                        ),
                        line,
                    }
                })
            }
            Condition::Compare { lhs, op, rhs } => {
                let negate = *op == CmpOp::NotEqual;
                let value = self.integer_of(rhs, line)?;
                if value < 0 {
                    return Err(parse_error(
                        line,
                        format!("negative integer in `if` condition is not supported: `{value}`"),
                    ));
                }
                let value = value as u64;
                if lhs.index.is_some() {
                    let bit = self.bit_of(lhs)?;
                    return Ok(match (value, negate) {
                        (0, false) | (1, true) => ClassicalCondition::BitIsZero(bit),
                        (1, false) | (0, true) => ClassicalCondition::BitIsOne(bit),
                        (other, _) => {
                            return Err(parse_error(
                                line,
                                format!("bit comparison must be against 0 or 1, got `{other}`"),
                            ));
                        }
                    });
                }
                let name = lhs
                    .register()
                    .ok_or_else(|| PrismError::UndefinedRegister {
                        name: lhs.describe(),
                        line,
                    })?;
                let register =
                    self.cregs
                        .get(name)
                        .ok_or_else(|| PrismError::UndefinedRegister {
                            name: name.to_string(),
                            line,
                        })?;
                Ok(if negate {
                    ClassicalCondition::RegisterNotEquals {
                        offset: register.offset,
                        size: register.size,
                        value,
                    }
                } else {
                    ClassicalCondition::RegisterEquals {
                        offset: register.offset,
                        size: register.size,
                        value,
                    }
                })
            }
        }
    }

    // --------------------------------------------------------------- pragmas

    /// Dispatch a `#pragma braket ...` line.
    ///
    /// Only the Braket dialect reads these. Under any other the pragma is
    /// rejected rather than dropped, so a lost result request cannot pass
    /// unnoticed.
    fn exec_pragma(&mut self, text: &str, line: usize) -> Result<Vec<Instruction>> {
        let text = text.strip_prefix('#').unwrap_or(text);
        let body = text.strip_prefix("pragma").unwrap_or(text).trim();
        // A pragma no pass here reads is ignored, as the language asks. One
        // naming `braket` is read, or declined, since dropping a noise or result
        // request this crate understands would change the answer.
        let Some(braket_body) = body
            .strip_prefix("braket")
            .filter(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
        else {
            return Ok(Vec::new());
        };
        if self.dialect != Dialect::Braket {
            return Err(PrismError::UnsupportedConstruct {
                construct: "`#pragma braket`, which needs `Dialect::Braket`".to_string(),
                line,
            });
        }
        if self.nested {
            return Err(PrismError::UnsupportedConstruct {
                construct: "`#pragma braket` inside a block body".to_string(),
                line,
            });
        }
        let braket_body = braket_body.trim();
        let (kind, rest) = match braket_body.split_once(char::is_whitespace) {
            Some((kind, rest)) => (kind, rest.trim()),
            None => (braket_body, ""),
        };
        match kind {
            "result" => {
                let total = self.total_qubits;
                let spec = braket::parse_result_pragma(rest, line, total, &|token| {
                    self.qubits_of_text(token, line)
                        .map(|targets| targets.to_vec())
                })?;
                self.results.push(spec);
                Ok(Vec::new())
            }
            "noise" => {
                let spec = braket::parse_noise_pragma(rest, line, &|token| {
                    self.qubits_of_text(token, line)
                        .map(|targets| targets.to_vec())
                })?;
                self.pending_noise = Some(spec);
                Ok(Vec::new())
            }
            "verbatim" => {
                self.verbatim_pending = true;
                Ok(Vec::new())
            }
            _ if braket_body.starts_with("unitary") => self.exec_unitary_pragma(braket_body, line),
            other => Err(PrismError::UnsupportedConstruct {
                construct: format!("`#pragma braket {other}`"),
                line,
            }),
        }
    }

    /// `#pragma braket unitary([[...]]) q[0]`, an inline gate matrix.
    fn exec_unitary_pragma(&mut self, body: &str, line: usize) -> Result<Vec<Instruction>> {
        let rest = body.trim_start_matches("unitary").trim();
        if !rest.starts_with('(') {
            return Err(parse_error(
                line,
                "`unitary` takes its matrix in parentheses",
            ));
        }
        let (matrix_text, targets_text) = braket::split_paren_body(rest, line)?;
        let matrix = braket::parse_matrix(matrix_text, line)?;
        let targets = self.qubits_of_text(targets_text, line)?;
        braket::unitary_instructions(&matrix, &targets, line)
    }

    /// Resolve a qubit reference written inside a pragma, whose body is its own
    /// grammar rather than part of the statement stream.
    pub(super) fn qubits_of_text(&self, token: &str, line: usize) -> Result<SmallVec<[usize; 4]>> {
        let mut out = SmallVec::new();
        for operand in syntax::parse_operands(&lexer::tokenize(token)?)? {
            out.extend(self.qubits_of(&Operand { line, ..operand })?);
        }
        Ok(out)
    }
}
