//! Walking a parsed program into instructions.
//!
//! Every name a statement reads is resolved against the scope the evaluator
//! holds, so a `for` body or a `gate` body is parsed once and run against
//! fresh bindings rather than rewritten and reparsed.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use super::*;
use crate::circuit::dynamic::{BinaryOp, ClassicalExpr, ExternFunction};
use crate::circuit::qasm::ast::{
    self, Argument, ArrayShape, AssignOp, Block, CmpOp, Condition, DefParam, ForRange, Index,
    Operand, OperandName, Stmt, StmtKind,
};
use crate::circuit::qasm::expr::{self as syntax_expr, Duration, Expr, Stretched, Timed};
use crate::circuit::qasm::stream::{MAX_NESTING, with_stack};
use crate::circuit::qasm::{lexer, parser as syntax};

mod calls;
mod control_flow;
mod declarations;
mod expressions;
mod operands;
mod pragmas;

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
        Timed::Stretch(_) => Err(parse_error(
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

/// `expr` with each array element it reads replaced by that element's value,
/// and each `sizeof` by the size it asks for.
fn resolve_elements<'e>(
    expr: &Expr<'e>,
    line: usize,
    vars: Option<&HashMap<&str, ClassicalValue>>,
    durations: &HashMap<&str, Timed>,
    arrays: &HashMap<&str, Rc<ClassicalArray>>,
) -> Result<Expr<'e>> {
    let fold = |index: &Expr| with_stack(|| number_of(index, line, vars, durations, arrays));
    replace_nodes(expr, &|node| match node {
        Expr::Element(element) => {
            let array = arrays.get(element.array).ok_or_else(|| {
                parse_error(line, format!("`{}` is not a declared array", element.array))
            })?;
            let at = element_offset(array, element.array, &element.indices, line, fold)?;
            Ok(Some(Expr::Number(array.values[at])))
        }
        Expr::Call(call) if call.name == "sizeof" => sizeof(call, line, arrays, &fold).map(Some),
        _ => Ok(None),
    })
}

/// `sizeof(a)` or `sizeof(a, d)`: the length of dimension `d`, `0` by default,
/// of an array or of the slice a subscript of it leaves.
fn sizeof<'e>(
    call: &syntax_expr::Call<'e>,
    line: usize,
    arrays: &HashMap<&str, Rc<ClassicalArray>>,
    fold: &impl Fn(&Expr) -> Result<ClassicalValue>,
) -> Result<Expr<'e>> {
    let (target, dimension) = match call.args.as_slice() {
        [target] => (target, 0),
        [target, dimension] => (target, whole(fold(dimension)?, line)?),
        _ => {
            return Err(parse_error(
                line,
                "`sizeof` takes an array and an optional dimension",
            ));
        }
    };
    let (name, indices) = match target {
        Expr::Ident(name) => (*name, &[][..]),
        Expr::Element(element) => (element.array, element.indices.as_slice()),
        _ => ("", &[][..]),
    };
    let array = arrays.get(name).ok_or_else(|| {
        parse_error(
            line,
            format!("`sizeof` takes a declared array, got `{target}`"),
        )
    })?;
    let view = array_view(array, name, indices, line, fold)?;
    let size = usize::try_from(dimension)
        .ok()
        .and_then(|dimension| view.dims.get(dimension))
        .ok_or_else(|| {
            parse_error(
                line,
                format!(
                    "`{target}` has {} dimension(s), so `sizeof` has no dimension {dimension}",
                    view.dims.len()
                ),
            )
        })?;
    Ok(Expr::Number(ClassicalValue::Int(*size as i128)))
}

/// A folded value read as a whole number.
fn whole(value: ClassicalValue, line: usize) -> Result<i64> {
    match value {
        ClassicalValue::Float(float) if float.fract() != 0.0 => Err(parse_error(
            line,
            format!("expected an integer, got {float}"),
        )),
        value => i64::try_from(value.to_int())
            .map_err(|_| parse_error(line, "integer expression out of range")),
    }
}

/// `expr` with each node `value` gives a replacement for replaced by it, and the
/// children of every other node visited in turn.
pub(super) fn replace_nodes<'e>(
    expr: &Expr<'e>,
    value: &impl Fn(&Expr<'e>) -> Result<Option<Expr<'e>>>,
) -> Result<Expr<'e>> {
    if let Some(replaced) = value(expr)? {
        return Ok(replaced);
    }
    let resolve = |inner: &Expr<'e>| with_stack(|| replace_nodes(inner, value));
    let bound = |inner: &Option<Expr<'e>>| inner.as_ref().map(resolve).transpose();
    Ok(match expr {
        Expr::Element(element) => Expr::Element(Box::new(syntax_expr::Element {
            array: element.array,
            indices: element.indices.iter().map(resolve).collect::<Result<_>>()?,
        })),
        Expr::Range(range) => Expr::Range(Box::new(ast::Range {
            start: bound(&range.start)?,
            step: bound(&range.step)?,
            stop: bound(&range.stop)?,
        })),
        Expr::Concat(parts) => Expr::Concat(parts.iter().map(resolve).collect::<Result<_>>()?),
        Expr::Set(entries) => Expr::Set(entries.iter().map(resolve).collect::<Result<_>>()?),
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
        at = at * size + position(fold(index)?.to_float(), size, name, line)?;
    }
    Ok(at)
}

/// The position index `written` names along a dimension of `size`, where a
/// negative index counts from the end.
fn position(written: f64, size: usize, name: &str, line: usize) -> Result<usize> {
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
    Ok(value as usize)
}

/// Positions an inclusive `start:step:stop` range names along a dimension of
/// `size`, an open bound reaching the end it faces.
fn range_positions(
    range: &ast::Range,
    size: usize,
    name: &str,
    line: usize,
    fold: impl Fn(&Expr) -> Result<ClassicalValue>,
) -> Result<Vec<usize>> {
    let bound = |expr: &Option<Expr>, fallback: i64| -> Result<i64> {
        match expr {
            None => Ok(fallback),
            Some(expr) => whole(fold(expr)?, line),
        }
    };
    let from_end = |value: i64| {
        if value < 0 {
            value.saturating_add(size as i64)
        } else {
            value
        }
    };
    let first = from_end(bound(&range.start, 0)?);
    let last = from_end(bound(&range.stop, size as i64 - 1)?);
    let step = bound(&range.step, 1)?;
    if step == 0 {
        return Err(parse_error(
            line,
            format!("range step in `{name}[{range}]` must be non-zero"),
        ));
    }
    let mut out = Vec::new();
    let mut at = first;
    while (step > 0 && at <= last) || (step < 0 && at >= last) {
        out.push(position(at as f64, size, name, line)?);
        let Some(next) = at.checked_add(step) else {
            break;
        };
        at = next;
    }
    Ok(out)
}

/// The elements a subscript of an array names: the shape it leaves, each fixed
/// index dropping its dimension and each range keeping one, and the row-major
/// positions it covers, in order.
struct ArrayView {
    dims: Vec<usize>,
    offsets: Vec<usize>,
}

fn array_view(
    array: &ClassicalArray,
    name: &str,
    indices: &[Expr],
    line: usize,
    fold: impl Fn(&Expr) -> Result<ClassicalValue>,
) -> Result<ArrayView> {
    if indices.len() > array.dims.len() {
        return Err(parse_error(
            line,
            format!(
                "`{name}` has {} dimension(s) but is indexed with {}",
                array.dims.len(),
                indices.len()
            ),
        ));
    }
    let mut dims = Vec::new();
    let mut offsets = vec![0usize];
    for (axis, &size) in array.dims.iter().enumerate() {
        let picks = match indices.get(axis) {
            None => (0..size).collect(),
            Some(Expr::Range(range)) => range_positions(range, size, name, line, &fold)?,
            Some(index) => vec![position(fold(index)?.to_float(), size, name, line)?],
        };
        if !matches!(indices.get(axis), Some(index) if !matches!(index, Expr::Range(_))) {
            dims.push(picks.len());
        }
        offsets = offsets
            .iter()
            .flat_map(|&base| picks.iter().map(move |&pick| base * size + pick))
            .collect();
    }
    Ok(ArrayView { dims, offsets })
}

/// An array value copied out of the arrays a program declares.
struct ArrayValue {
    dims: Vec<usize>,
    values: Vec<ClassicalValue>,
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

/// An array argument written with a call's operand grammar, `a` or `a[1:3]`,
/// as the expression that names it.
fn operand_reference<'e>(operand: &Operand<'e>) -> Result<Expr<'e>> {
    let Some(name) = operand.register() else {
        return Err(parse_error(
            operand.line,
            format!("`{}` is a qubit where an array belongs", operand.describe()),
        ));
    };
    let index = match &operand.index {
        None => return Ok(Expr::Ident(name)),
        Some(Index::Single(index)) => index.clone(),
        Some(Index::Range(range)) => Expr::Range(range.clone()),
        Some(Index::Set(_)) => {
            return Err(parse_error(
                operand.line,
                format!(
                    "`{}` indexes an array with a set, which the language does not allow",
                    operand.describe()
                ),
            ));
        }
    };
    Ok(Expr::Element(Box::new(syntax_expr::Element {
        array: name,
        indices: vec![index],
    })))
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

fn non_affine(expr: &Expr, line: usize) -> PrismError {
    unsupported(
        line,
        format!("the angle `{expr}`, which is not affine in the `input` it reads"),
    )
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
        let mut instructions = self.execute(&program)?;
        if self.verbatim_pending {
            return Err(parse_error(
                program.last().map_or(1, |stmt| stmt.line),
                "a verbatim pragma must be followed by a `box`",
            ));
        }
        self.check_supplied()?;
        let links = self.link_tags(&mut instructions);

        let circuit = Circuit {
            num_qubits: self.total_qubits,
            num_classical_bits: self.total_cbits,
            instructions,
        };
        let params = Parameters::from_links(links, self.input_names.len())
            .with_names(std::mem::take(&mut self.input_names))
            .pinned_to(&circuit);
        let noise = self.build_noise_model(&circuit)?;
        Ok((circuit, params, std::mem::take(&mut self.results), noise))
    }

    /// Link each gate holding a tag to the slot it reads, leaving the angle the
    /// slot gives it at a value of zero in its place.
    fn link_tags(&self, instructions: &mut [Instruction]) -> Vec<ParamLink> {
        let tags = self.tags.borrow();
        let mut links = Vec::new();
        for (at, instr) in instructions.iter_mut().enumerate() {
            let Some(id) = gate_angle(instr).and_then(tag_id) else {
                continue;
            };
            let form = tags[id];
            let slot = form.slot.expect("a tag reads a slot");
            *angle_mut(instr) = form.offset;
            links.push(ParamLink {
                instruction: at,
                slot,
                scale: form.scale,
                offset: form.offset,
            });
        }
        links
    }

    /// Take the caller's `input` values, which each declaration reads as it runs.
    pub(super) fn supply(&mut self, inputs: &[(&str, ClassicalValue)]) -> Result<()> {
        for &(name, value) in inputs {
            if self.supplied.iter().any(|(seen, _)| seen == name) {
                return Err(PrismError::InvalidParameter {
                    message: format!("input `{name}` is supplied twice"),
                });
            }
            self.supplied.push((name.to_string(), Some(value)));
        }
        Ok(())
    }

    /// Take the caller's `extern` functions, which each declaration binds as it
    /// runs.
    pub(super) fn link_externs(&mut self, externs: &Externs) {
        self.registry = Some(Rc::new(externs.functions.clone()));
    }

    /// Bind an `extern` declaration to the function registered under its name.
    fn declare_extern(
        &mut self,
        name: &'a str,
        params: &[ast::DefType<'a>],
        result: Option<&ast::DefType<'a>>,
        line: usize,
    ) -> Result<()> {
        if self.nested {
            return Err(parse_error(
                line,
                format!("extern `{name}` is declared outside the global scope"),
            ));
        }
        if self.gate_defs.contains_key(name)
            || self.def_defs.contains_key(name)
            || self.externs.contains_key(name)
        {
            return Err(parse_error(line, format!("`{name}` is already declared")));
        }
        self.reject_redeclaration(name, line)?;
        let Some(registry) = self.registry.as_ref() else {
            return Err(unsupported(
                line,
                format!(
                    "extern `{name}`, which needs a function from a `_with_externs` entry point"
                ),
            ));
        };
        let registry = Rc::clone(registry);
        let Some(result) = result else {
            return Err(unsupported(
                line,
                format!("extern `{name}` with no result type"),
            ));
        };
        let params = params
            .iter()
            .map(|param| self.extern_type(name, param, line))
            .collect::<Result<Vec<_>>>()?;
        let result = self.extern_type(name, result, line)?;
        let Some(callback) = registry.get(name) else {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "extern `{name}` has no function registered under that name; register one \
                     in `Externs` and parse with a `_with_externs` entry point"
                ),
            });
        };
        let function = ExternFunction {
            name: name.to_string(),
            params,
            result,
            callback: Arc::clone(callback),
        };
        Rc::make_mut(&mut self.externs).insert(name, Arc::new(function));
        Ok(())
    }

    fn extern_type(&self, name: &str, ty: &ast::DefType, line: usize) -> Result<RuntimeType> {
        match (ty.ty, &ty.width) {
            ("bit", None) => Ok(RuntimeType::Bool),
            ("bit", Some(width)) => match self.integer_of(width, line)? {
                bits @ 1..=64 => Ok(RuntimeType::Uint { width: bits as u32 }),
                bits if bits < 1 => Err(parse_error(line, format!("`bit[{bits}]` has no bits"))),
                bits => Err(unsupported(
                    line,
                    format!(
                        "extern `{name}` over a `bit[{bits}]` value, past the 64 bits a value \
                         holds"
                    ),
                )),
            },
            ("bool" | "int" | "uint" | "float", _) | ("angle", None) => {
                self.runtime_type(ty.ty, ty.width.as_ref(), line)
            }
            (spelled, width) => Err(unsupported(
                line,
                match width {
                    Some(width) => format!("extern `{name}` over a `{spelled}[{width}]` value"),
                    None => format!("extern `{name}` over a `{spelled}` value"),
                },
            )),
        }
    }

    /// Reject a call to `function` with the wrong number of arguments.
    pub(super) fn check_extern_arity(
        function: &ExternFunction,
        args: usize,
        line: usize,
    ) -> Result<()> {
        if args != function.params.len() {
            return Err(parse_error(
                line,
                format!(
                    "extern `{}` takes {} argument(s), got {args}",
                    function.name,
                    function.params.len()
                ),
            ));
        }
        Ok(())
    }

    /// `expr` with every `extern` call it makes replaced by the value the call
    /// returns, each argument folded first.
    fn fold_externs<'x, 'e>(&self, expr: &'x Expr<'e>, line: usize) -> Result<Cow<'x, Expr<'e>>> {
        if self.externs.is_empty()
            || expr
                .finds_call(&|name| self.externs.contains_key(name))
                .is_none()
        {
            return Ok(Cow::Borrowed(expr));
        }
        replace_nodes(expr, &|node| {
            let Expr::Call(call) = node else {
                return Ok(None);
            };
            let Some(function) = self.externs.get(call.name) else {
                return Ok(None);
            };
            Self::check_extern_arity(function, call.args.len(), line)?;
            let args = call
                .args
                .iter()
                .map(|arg| self.typed_of(arg, line))
                .collect::<Result<SmallVec<[ClassicalValue; 4]>>>()?;
            Ok(Some(Expr::Number(function.call(&args)?)))
        })
        .map(Cow::Owned)
    }

    /// Reject a supplied value that no `input` declaration took.
    pub(super) fn check_supplied(&self) -> Result<()> {
        match self.supplied.iter().find(|(_, value)| value.is_some()) {
            Some((name, _)) => Err(PrismError::InvalidParameter {
                message: format!("a value is supplied for `{name}`, which no `input` declares"),
            }),
            None => Ok(()),
        }
    }

    /// Allocate the tag that stands for `form` in an angle.
    fn tag(&self, form: Linear) -> f64 {
        let mut tags = self.tags.borrow_mut();
        tags.push(form);
        f64::from_bits(TAG | (tags.len() - 1) as u64)
    }

    /// The affine form `name` holds when its value is a tag.
    fn symbol(&self, name: &str) -> Option<Linear> {
        match self.param_vars.as_ref()?.get(name)? {
            ClassicalValue::Float(value) => tag_id(*value).map(|id| self.tags.borrow()[id]),
            _ => None,
        }
    }

    /// A name `expr` reads whose value is a tag.
    fn symbol_in(&self, expr: &Expr) -> Option<&'a str> {
        self.symbolic
            .iter()
            .copied()
            .find(|name| expr.mentions(name) && self.symbol(name).is_some())
    }

    fn symbol_error(&self, name: &str, place: &str, line: usize) -> PrismError {
        let what = if self.inputs.contains_key(name) {
            format!("input `{name}`")
        } else {
            format!("`{name}`, which an `input` reaches,")
        };
        unsupported(line, format!("{what} {place}"))
    }

    /// Decline `expr` when it reads a tagged name, which only a gate angle may.
    fn reject_symbol(&self, expr: &Expr, line: usize) -> Result<()> {
        match self.symbol_in(expr) {
            Some(name) => Err(self.symbol_error(name, "outside a gate angle", line)),
            None => Ok(()),
        }
    }

    /// Run a block, threading the noise event each statement may leave behind
    /// onto the instruction index it belongs to.
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
            if let Err(err) = self.exec_stmt(stmt, &mut instructions)
                && !self.unwinds_end()
            {
                return Err(err);
            }
            // Nested blocks charged what they emitted as they ran; this
            // statement owes only what it added on top.
            let nested = left - self.expansion_left.get();
            self.charge(
                (instructions.len() - base).saturating_sub(nested).max(1),
                stmt.line,
            )?;
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
            if self.exit.is_some()
                || self
                    .def_result
                    .as_ref()
                    .is_some_and(|result| result.returned)
            {
                break;
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
                self.resolve_calls(value, line, out)?;
                Ok(())
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
                if *kind == ast::RegisterKind::Qubit && self.def_result.is_some() {
                    return Err(parse_error(
                        line,
                        "qubit declarations are only allowed at global scope",
                    ));
                }
                if let Some(result) = self.def_result.as_ref().filter(|result| {
                    *kind == ast::RegisterKind::Classical && result.local == Some(*name)
                }) {
                    let runtime = result.runtime;
                    let bits = result.bits.clone();
                    if init.is_some() && !runtime {
                        return Err(unsupported(
                            line,
                            format!(
                                "an initializer on `{name}`, the bit def returns; the \
                                 caller's bits hold it and the instruction list has no \
                                 classical write"
                            ),
                        ));
                    }
                    // The returned bit is the caller's target, bound before the
                    // body ran, so its declaration allocates nothing.
                    let width = match size {
                        Some(size) => self.integer_of(size, line)?,
                        None => 1,
                    };
                    if width != bits.len() as i64 {
                        return Err(parse_error(
                            line,
                            format!(
                                "`{name}` declares {width} bit(s) where the def returns {}",
                                bits.len()
                            ),
                        ));
                    }
                    if runtime {
                        if let Some(init) = init {
                            self.init_bits(name, size.as_ref(), init, line)?;
                        } else {
                            for bits in bits.chunks(64) {
                                self.write_classical_bits(
                                    bits,
                                    ClassicalExpr::Const(ClassicalValue::Int(0)),
                                    line,
                                )?;
                            }
                        }
                    }
                    return Ok(());
                }
                if self.in_while() {
                    return Err(unsupported(
                        line,
                        format!(
                            "register `{name}` declared inside a `while` loop; declare it \
                             before the loop"
                        ),
                    ));
                }
                if *kind == ast::RegisterKind::Classical
                    && self.def_result.is_some()
                    && self.private_bits.is_some()
                {
                    return self.declare_private(name, size.as_ref(), init.as_deref(), line);
                }
                self.declare_register(*kind, name, size.as_ref(), line)?;
                if let Some(init) = init {
                    self.init_bits(name, size.as_ref(), init, line)?;
                }
                Ok(())
            }
            StmtKind::InputDecl { ty, name, size } => {
                self.declare_input(stmt, ty, size.as_ref(), name, line)
            }
            StmtKind::OutputDecl { ty, name, size } => {
                if *ty != "bit" {
                    return Err(unsupported(
                        line,
                        format!("`output {ty}` (only bit outputs are reported)"),
                    ));
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
                let value = value
                    .as_ref()
                    .map(|value| self.resolve_calls(value, line, out))
                    .transpose()?;
                if self.is_runtime_decl(stmt)
                    || value
                        .as_ref()
                        .is_some_and(|value| self.reads_runtime(value))
                {
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
                let value = self.resolve_calls(value, line, out)?;
                let target = Operand {
                    name: OperandName::Register(target),
                    index: None,
                    line,
                };
                self.assign_bits(&target, *op, &value, line)
            }
            StmtKind::Assign { target, op, value } => {
                let value = self.resolve_calls(value, line, out)?;
                self.assign_classical(target, *op, &value, line)?;
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
                    return Err(unsupported(
                        line,
                        format!(
                            "a measurement into variable `{name}`; measure into a bit and \
                             assign the bit"
                        ),
                    ));
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
                let instructions = self.exec_if(conditional, line, out)?;
                out.extend(instructions);
                Ok(())
            }
            StmtKind::For {
                ty,
                width,
                variable,
                range,
                body,
            } => {
                let ty = self.classical_type(ty, width.as_ref(), line)?;
                self.exec_for(stmt, variable, ty, range, body, line, out)
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
            StmtKind::While { .. } => Err(unsupported(line, "while")),
            StmtKind::Break => self.exec_loop_exit(Exit::Break, line),
            StmtKind::Continue => self.exec_loop_exit(Exit::Continue, line),
            StmtKind::End => self.exec_end(line),
            StmtKind::GateDef {
                name,
                params,
                qubits,
                body,
            } => {
                self.reject_extern_name(name, line)?;
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
            StmtKind::ExternDecl {
                name,
                params,
                result,
            } => self.declare_extern(name, params, result.as_ref(), line),
            StmtKind::Return(value) => self.exec_return(value.as_ref(), line, out),
            StmtKind::CallAssign(assign) => with_stack(|| self.exec_call_assign(assign, line, out)),
            StmtKind::ArrayDecl(decl) if self.is_runtime_decl(stmt) => Err(unsupported(
                line,
                format!(
                    "array `{}`, written where a loop or a measurement decides the value",
                    decl.name
                ),
            )),
            StmtKind::ArrayDecl(decl) => self.declare_array(decl, line),
            StmtKind::ElementAssign(assign) if self.is_bit_name(assign.array) => {
                let value = self.resolve_calls(&assign.value, line, out)?;
                let [index] = assign.indices.as_slice() else {
                    return Err(parse_error(
                        line,
                        format!("`{}` holds bits and takes one index", assign.array),
                    ));
                };
                let index = match index {
                    Expr::Range(range) => Index::Range(range.clone()),
                    index => Index::Single(index.clone()),
                };
                let target = Operand {
                    name: OperandName::Register(assign.array),
                    index: Some(index),
                    line,
                };
                self.assign_bits(&target, assign.op, &value, line)
            }
            StmtKind::ElementAssign(assign) => {
                let mut resolved = assign.as_ref().clone();
                resolved.value = self.resolve_calls(&assign.value, line, out)?;
                if self
                    .dynamic
                    .as_ref()
                    .is_some_and(|state| state.bit_values.contains_key(assign.array))
                {
                    return self.assign_bit_parameter(&resolved, line);
                }
                self.assign_element(&resolved, line)
            }
            StmtKind::Box { duration, body } => {
                // A box and Braket's verbatim pragma both direct a scheduler or
                // a device compiler, which a simulator has nothing to honour.
                self.verbatim_pending = false;
                let length = match duration {
                    Some(duration) => self.span_of(duration, "box", line)?,
                    None => None,
                };
                out.extend(self.exec_box(body)?);
                if let Some(length) = length {
                    self.resolve_stretches(body, length, line)?;
                }
                Ok(())
            }
            StmtKind::Pragma(text) => {
                out.extend(self.exec_pragma(text, line)?);
                Ok(())
            }
        }
    }
}
