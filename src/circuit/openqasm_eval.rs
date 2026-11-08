//! Walking a parsed program into instructions.
//!
//! Every name a statement reads is resolved against the scope the evaluator
//! holds, so a `for` body or a `gate` body is parsed once and run against
//! fresh bindings rather than rewritten and reparsed.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use super::*;
use crate::circuit::dynamic::{ClassicalExpr, ExternFunction};
use crate::circuit::qasm::ast::{
    self, Argument, ArrayShape, AssignOp, Block, CmpOp, Condition, DefParam, ForRange, Index,
    Operand, OperandName, Stmt, StmtKind,
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
fn replace_nodes<'e>(
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
    PrismError::UnsupportedConstruct {
        construct: format!("the angle `{expr}`, which is not affine in the `input` it reads"),
        line,
    }
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
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "extern `{name}`, which needs a function from a `_with_externs` entry point"
                ),
                line,
            });
        };
        let registry = Rc::clone(registry);
        let Some(result) = result else {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("extern `{name}` with no result type"),
                line,
            });
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
            ("bool" | "int" | "uint" | "float", _) | ("angle", None) => {
                self.runtime_type(ty.ty, ty.width.as_ref(), line)
            }
            (spelled, width) => Err(PrismError::UnsupportedConstruct {
                construct: match width {
                    Some(width) => format!("extern `{name}` over a `{spelled}[{width}]` value"),
                    None => format!("extern `{name}` over a `{spelled}` value"),
                },
                line,
            }),
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
        PrismError::UnsupportedConstruct {
            construct: format!("{what} {place}"),
            line,
        }
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
            StmtKind::InputDecl { ty, name, size } => {
                self.declare_input(stmt, ty, size.as_ref(), name, line)
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
                let instructions = self.exec_if(conditional, line, out)?;
                out.extend(instructions);
                Ok(())
            }
            StmtKind::For {
                variable,
                range,
                body,
            } => self.exec_for(stmt, variable, range, body, line, out),
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
            StmtKind::While { .. } => Err(PrismError::UnsupportedConstruct {
                construct: "while".to_string(),
                line,
            }),
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

    /// Declare an `input`: a supplied one as a variable holding its value, an
    /// unsupplied `float` or `angle` as the next slot, read through a tag.
    fn declare_input(
        &mut self,
        stmt: &Stmt<'a>,
        ty: &'a str,
        size: Option<&Expr<'a>>,
        name: &'a str,
        line: usize,
    ) -> Result<()> {
        if self.inputs.contains_key(name) {
            return Err(parse_error(line, format!("input `{name}` declared twice")));
        }
        let supplied = self
            .supplied
            .iter_mut()
            .find(|(supplied, _)| supplied == name)
            .and_then(|(_, value)| value.take());
        if let Some(value) = supplied {
            let value = Expr::Number(value);
            if ty == "bit" {
                self.declare_register(ast::RegisterKind::Classical, name, size, line)?;
                return self.init_bits(name, size, &ast::BitInit::Value(value), line);
            }
            if self.is_runtime_decl(stmt) {
                return self.declare_runtime(false, ty, size, name, Some(&value), line);
            }
            return self.declare_classical(false, ty, size, name, Some(&value), line);
        }
        if !matches!(ty, "float" | "angle") {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "input `{name}` is `{ty}`, which binds no gate angle; supply its value"
                ),
            });
        }
        if self.dynamic.is_some() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "unsupplied input `{name}` in a dynamic program, which has no slot to bind"
                ),
                line,
            });
        }
        let slot = self.input_names.len();
        self.inputs.insert(name, slot);
        self.input_names.push(name.to_string());
        let tag = self.tag(Linear {
            slot: Some(slot),
            scale: 1.0,
            offset: 0.0,
        });
        self.param_vars
            .get_or_insert_with(HashMap::new)
            .insert(name, ClassicalValue::Float(tag));
        self.symbolic.push(name);
        Ok(())
    }

    pub(super) fn classical_type(
        &self,
        ty: &str,
        width: Option<&Expr>,
        line: usize,
    ) -> Result<ClassicalType> {
        Ok(match ty {
            "int" | "uint" => ClassicalType::Int(self.runtime_type(ty, width, line)?),
            "bool" => ClassicalType::Bool,
            "float" | "angle" => self.real_type(ty, width, line)?,
            "complex" => match self.real_type("float", width, line)? {
                ClassicalType::Float(bits) => ClassicalType::Complex(bits),
                other => other,
            },
            "duration" => ClassicalType::Duration,
            "stretch" => ClassicalType::Stretch,
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`{other}` declarations"),
                    line,
                });
            }
        })
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
        let kind = self.classical_type(ty, width, line)?;
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
            } else if let ClassicalType::Complex(_) = kind {
                self.bind_complex(name, kind, Complex64::new(0.0, 0.0), constant);
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
        if let ClassicalType::Complex(_) = kind {
            let folded = self.complex_of(value, line)?;
            self.bind_complex(name, kind, folded, constant);
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
        let decl = ClassicalDecl {
            ty,
            constant,
            value: constant.then_some(ConstValue::Duration(value)),
        };
        if Rc::make_mut(&mut self.classical)
            .insert(name, decl)
            .is_none()
        {
            self.declared.push(name);
        }
        self.durations.insert(name, value);
    }

    /// Bind a `complex[float[n]]`, rounding each component to the declared width.
    fn bind_complex(&mut self, name: &'a str, ty: ClassicalType, value: Complex64, constant: bool) {
        let value = match ty {
            ClassicalType::Complex(width) => Complex64::new(
                crate::circuit::dynamic::real(value.re, width),
                crate::circuit::dynamic::real(value.im, width),
            ),
            _ => value,
        };
        let decl = ClassicalDecl {
            ty,
            constant,
            value: constant.then_some(ConstValue::Complex(value)),
        };
        if Rc::make_mut(&mut self.classical)
            .insert(name, decl)
            .is_none()
        {
            self.declared.push(name);
        }
        self.complex.insert(name, value);
    }

    /// Fold an expression to the complex value it has. A real one is promoted.
    fn complex_of(&self, expr: &Expr, line: usize) -> Result<Complex64> {
        if self.reads_runtime(expr) {
            return Err(super::dynamic::runtime_complex(line));
        }
        if let Some(name) = self.symbol_in(expr) {
            return Err(self.symbol_error(name, "in a `complex` value", line));
        }
        let vars = self.param_vars.as_ref();
        let expr = self.with_complex(expr)?;
        let expr = if !self.arrays.is_empty() && expr.reads_element() {
            Cow::Owned(resolve_elements(
                &expr,
                line,
                vars,
                &self.durations,
                &self.arrays,
            )?)
        } else {
            expr
        };
        syntax_expr::eval_complex(&expr, line, vars)
    }

    /// `expr` with each `complex` name it reads replaced by its value.
    fn with_complex<'x, 'e>(&self, expr: &'x Expr<'e>) -> Result<Cow<'x, Expr<'e>>> {
        if self.complex.is_empty() || !self.complex.keys().any(|name| expr.mentions(name)) {
            return Ok(Cow::Borrowed(expr));
        }
        replace_nodes(expr, &|node| {
            Ok(match node {
                Expr::Ident(name) => self.complex.get(name).map(|value| Expr::Complex(*value)),
                _ => None,
            })
        })
        .map(Cow::Owned)
    }

    /// Fold an expression that has to be a duration or a stretch.
    fn duration_of(&self, expr: &Expr, line: usize) -> Result<Timed> {
        self.reject_symbol(expr, line)?;
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
        if self.arrays.contains_key(target) {
            let assign = ast::ElementAssign {
                array: target,
                indices: Vec::new(),
                op,
                value: value.clone(),
            };
            return self.assign_element(&assign, line);
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
        if let ClassicalType::Complex(_) = kind {
            let folded = self.complex_of(value, line)?;
            let updated = match op {
                None => folded,
                Some(op) => {
                    let current = self.complex.get(target).copied().unwrap_or_default();
                    let combined = Expr::Binary {
                        op: op.binary(),
                        left: Box::new(Expr::Complex(current)),
                        right: Box::new(Expr::Complex(folded)),
                    };
                    syntax_expr::eval_complex(&combined, line, None)?
                }
            };
            self.bind_complex(target, kind, updated, false);
            return Ok(());
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

    /// The element type of an `array[ty[width], ...]`.
    fn array_kind(&self, ty: &str, width: Option<&Expr>, line: usize) -> Result<ClassicalType> {
        match ty {
            "int" | "uint" | "bool" | "float" | "angle" => self.classical_type(ty, width, line),
            other => Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`array[{other}, ...]`; an array holds `int`, `uint`, `float`, \
                     `angle` or `bool`, and bits belong in a `bit[n]` register"
                ),
                line,
            }),
        }
    }

    /// The array `expr` names, copied out: a whole array, a slice or a
    /// partial subscript of one, or arrays joined with `++` along their first
    /// dimension. `None` when `expr` is a single value.
    fn array_value(&self, expr: &Expr, line: usize) -> Result<Option<ArrayValue>> {
        with_stack(|| match expr {
            Expr::Ident(name) => Ok(self.arrays.get(name).map(|array| ArrayValue {
                dims: array.dims.clone(),
                values: array.values.clone(),
            })),
            Expr::Element(element) => {
                let Some(array) = self.arrays.get(element.array) else {
                    return Ok(None);
                };
                let view = array_view(array, element.array, &element.indices, line, |index| {
                    self.typed_of(index, line)
                })?;
                if view.dims.is_empty() {
                    return Ok(None);
                }
                Ok(Some(ArrayValue {
                    dims: view.dims,
                    values: view.offsets.iter().map(|&at| array.values[at]).collect(),
                }))
            }
            Expr::Concat(parts) => {
                let mut joined: Option<ArrayValue> = None;
                for part in parts {
                    let Some(next) = self.array_value(part, line)? else {
                        return Err(parse_error(
                            line,
                            format!("`{part}` is not an array, so `++` cannot join it"),
                        ));
                    };
                    match joined.as_mut() {
                        None => joined = Some(next),
                        Some(acc) if acc.dims[1..] == next.dims[1..] => {
                            acc.dims[0] += next.dims[0];
                            acc.values.extend(next.values);
                        }
                        Some(acc) => {
                            return Err(parse_error(
                                line,
                                format!(
                                    "`++` joins arrays of shapes {:?} and {:?}, whose inner \
                                     dimensions differ",
                                    acc.dims, next.dims
                                ),
                            ));
                        }
                    }
                }
                Ok(joined)
            }
            _ => Ok(None),
        })
    }

    fn declare_array(&mut self, decl: &ast::ArrayDecl<'a>, line: usize) -> Result<()> {
        let kind = self.array_kind(decl.ty, decl.width.as_ref(), line)?;
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
            (ast::ArrayInit::Value(expr), Some(_)) => {
                let Some(source) = self.array_value(expr, line)? else {
                    return Err(parse_error(
                        line,
                        format!("initializer for `{name}` does not match its dimensions"),
                    ));
                };
                if source.dims != dims {
                    return Err(parse_error(
                        line,
                        format!(
                            "initializer `{expr}` for `{name}` has shape {:?} where a shape of \
                             {dims:?} stands",
                            source.dims
                        ),
                    ));
                }
                values.extend(source.values.into_iter().map(|value| kind.store(value)));
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
        if assign.indices.len() < array.dims.len()
            || assign
                .indices
                .iter()
                .any(|index| matches!(index, Expr::Range(_)))
        {
            return self.assign_slice(assign, line);
        }
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

    /// Copy an array value into the elements a slice or a partial subscript
    /// names, which must have the shape the value has.
    fn assign_slice(&mut self, assign: &ast::ElementAssign<'a>, line: usize) -> Result<()> {
        let array = &self.arrays[assign.array];
        let kind = array.ty;
        let view = array_view(array, assign.array, &assign.indices, line, |index| {
            self.typed_of(index, line)
        })?;
        let target = if assign.indices.is_empty() {
            assign.array.to_string()
        } else {
            Expr::Element(Box::new(syntax_expr::Element {
                array: assign.array,
                indices: assign.indices.clone(),
            }))
            .to_string()
        };
        if assign.op.is_some() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("a compound assignment to the array slice `{target}`"),
                line,
            });
        }
        let Some(source) = self.array_value(&assign.value, line)? else {
            return Err(parse_error(
                line,
                format!(
                    "`{}` is not an array, so it cannot fill `{target}`, whose shape is {:?}",
                    assign.value, view.dims
                ),
            ));
        };
        if source.dims != view.dims {
            return Err(parse_error(
                line,
                format!(
                    "shape mismatch: `{target}` has shape {:?} and `{}` has shape {:?}",
                    view.dims, assign.value, source.dims
                ),
            ));
        }
        if let Some(array) = self.arrays.get_mut(assign.array) {
            let array = Rc::make_mut(array);
            for (&at, value) in view.offsets.iter().zip(source.values) {
                array.values[at] = kind.store(value);
            }
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
        self.reject_symbol(value, line)?;
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

    /// Fold an initializer to the value its declared type holds. An integer or
    /// `bool` takes a whole number, which a fractional float is not.
    fn fold_typed(&self, kind: ClassicalType, value: &Expr, line: usize) -> Result<ClassicalValue> {
        if let Some(name) = self.symbol_in(value) {
            return Err(self.symbol_error(
                name,
                "in a classical value; an unsupplied input binds only a gate angle",
                line,
            ));
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
        result: Option<&ast::DefType<'a>>,
        body: &Block<'a>,
        line: usize,
    ) -> Result<()> {
        self.reject_extern_name(name, line)?;
        let mut args = args.to_vec();
        // Freeze signature widths before a caller can shadow their bindings.
        for param in &mut args {
            if let DefParam::Array { width, shape, .. } = param {
                let sizes = match shape {
                    ArrayShape::Sized(dims) => dims.iter_mut().collect::<Vec<_>>(),
                    ArrayShape::Rank(rank) => vec![rank],
                };
                for size in sizes.into_iter().chain(width.as_mut()) {
                    let value = i128::from(self.integer_of(size, line)?);
                    *size = Expr::Number(ClassicalValue::Int(value));
                }
            }
            if let DefParam::Qubit {
                width: Some(width), ..
            }
            | DefParam::Bit {
                width: Some(width), ..
            }
            | DefParam::Value {
                width: Some(width), ..
            } = param
            {
                let value = i128::from(self.integer_of(width, line)?);
                *width = Expr::Number(ClassicalValue::Int(value));
            }
        }
        for param in &args {
            match param {
                DefParam::Value { ty, width, .. } => {
                    self.classical_type(ty, width.as_ref(), line)?;
                }
                DefParam::Array { ty, width, .. } => {
                    self.array_kind(ty, width.as_ref(), line)?;
                }
                _ => {}
            }
        }
        let mut result = result.cloned();
        if let Some(width) = result.as_mut().and_then(|result| result.width.as_mut()) {
            let value = i128::from(self.integer_of(width, line)?);
            *width = Expr::Number(ClassicalValue::Int(value));
        }
        if let Some(result) = result.as_ref().filter(|result| result.ty != "bit") {
            self.classical_type(result.ty, result.width.as_ref(), line)?;
        }
        self.charge(self.arrays.len(), line)?;
        let arrays = self
            .arrays
            .iter()
            .filter(|(_, array)| array.constant)
            .map(|(name, array)| (*name, Rc::clone(array)))
            .collect();
        let constants = Rc::clone(&self.classical);
        let Some((last, rest)) = body.split_last() else {
            if result.is_some() {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares a result but its body is empty"),
                ));
            }
            Rc::make_mut(&mut self.def_defs).insert(
                name,
                DefDefinition {
                    args,
                    result: None,
                    constants,
                    arrays,
                    local: None,
                    body: Block::new(),
                },
            );
            return Ok(());
        };
        let early = return_line(rest).or_else(|| match last.kind {
            StmtKind::Return(_) => None,
            _ => return_line(std::slice::from_ref(last)),
        });
        let computed_bit = self.dynamic.is_some()
            && matches!(&last.kind, StmtKind::Return(Some(ast::ReturnValue::Value(value)))
                if value.as_ident().is_none_or(|name| !rest.iter().any(|stmt| matches!(&stmt.kind,
                    StmtKind::RegisterDecl { kind: ast::RegisterKind::Classical, name: local, .. } if *local == name))));
        let local = if result.as_ref().is_some_and(|result| result.ty == "bit")
            && early.is_none()
            && !computed_bit
        {
            Self::def_result_local(name, &args, true, last, rest)?
        } else {
            Self::check_returns(name, result.as_ref(), body)?;
            if result.is_some() && return_line(body).is_none() {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares a result but does not end by returning one"),
                ));
            }
            None
        };
        Self::check_def_writes(name, &args, local, body, true)?;
        Rc::make_mut(&mut self.def_defs).insert(
            name,
            DefDefinition {
                args,
                result,
                constants,
                arrays,
                local,
                body: body.clone(),
            },
        );
        Ok(())
    }

    fn reject_extern_name(&self, name: &str, line: usize) -> Result<()> {
        if self.externs.contains_key(name) {
            return Err(parse_error(line, format!("`{name}` is already an extern")));
        }
        Ok(())
    }

    fn check_returns(name: &str, result: Option<&ast::DefType>, body: &[Stmt]) -> Result<()> {
        with_stack(|| {
            for stmt in body {
                match &stmt.kind {
                    StmtKind::Return(value) if value.is_some() != result.is_some() => {
                        return Err(parse_error(
                            stmt.line,
                            format!(
                                "def `{name}` returns a value inconsistent with its declared result"
                            ),
                        ));
                    }
                    StmtKind::If(branch) => {
                        Self::check_returns(name, result, &branch.then_body)?;
                        if let Some(body) = &branch.else_body {
                            Self::check_returns(name, result, body)?;
                        }
                    }
                    StmtKind::For { body, .. }
                    | StmtKind::While { body, .. }
                    | StmtKind::Box { body, .. } => Self::check_returns(name, result, body)?,
                    StmtKind::Switch { arms, .. } => {
                        for arm in arms {
                            Self::check_returns(name, result, &arm.body)?;
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        })
    }

    fn exec_return(
        &mut self,
        value: Option<&ast::ReturnValue<'a>>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let Some(frame) = &self.def_result else {
            return Err(PrismError::UnsupportedConstruct {
                construct: "`return` outside a def".to_string(),
                line,
            });
        };
        if self.guard.is_some() {
            return Err(PrismError::UnsupportedConstruct {
                construct: "a measurement-dependent return from a def".to_string(),
                line,
            });
        }
        if frame.runtime {
            return self.exec_runtime_return(value, line, out);
        }
        let ty = frame.ty;
        let returned = match (value, ty) {
            (Some(ast::ReturnValue::Measure(source)), ty) => {
                if ty.is_some() {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: "a measurement returned as a classical scalar".to_string(),
                        line,
                    });
                }
                let qubits = self.qubits_of(source)?;
                out.extend(Self::build_measurements(qubits, frame.bits.clone(), line)?);
                None
            }
            (Some(ast::ReturnValue::Value(value)), Some(ty)) => {
                let value = self.resolve_calls(value, line, out)?;
                if let ClassicalType::Complex(width) = ty {
                    let value = self.complex_of(&value, line)?;
                    let frame = self.def_result.as_mut().expect("checked above");
                    frame.value = Some(Expr::Complex(Complex64::new(
                        crate::circuit::dynamic::real(value.re, width),
                        crate::circuit::dynamic::real(value.im, width),
                    )));
                    frame.returned = true;
                    return Ok(());
                }
                // Folded once here, so the runtime check below does not call an
                // extern that the fold then calls again.
                let value = if self.reads_runtime(&value) {
                    value
                } else {
                    self.fold_externs(&value, line)?.into_owned()
                };
                if ty != ClassicalType::Duration
                    && self
                        .lower_expr(&value, line)
                        .is_ok_and(|value| !matches!(value, ClassicalExpr::Const(_)))
                {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: "a measurement-dependent scalar returned from a def".to_string(),
                        line,
                    });
                }
                Some(if ty == ClassicalType::Duration {
                    match self.duration_of(&value, line)? {
                        Timed::Duration(duration) => Expr::Duration(duration),
                        Timed::Stretch => {
                            return Err(PrismError::UnsupportedConstruct {
                                construct: "a stretch returned as a duration".to_string(),
                                line,
                            });
                        }
                        Timed::Number(_) => unreachable!("duration_of rejects numbers"),
                    }
                } else {
                    Expr::Number(ty.store(self.fold_typed(ty, &value, line)?))
                })
            }
            (Some(ast::ReturnValue::Value(value)), None) if frame.local.is_none() => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("return of `{value}` as a bit without a classical move"),
                    line,
                });
            }
            _ => None,
        };
        let frame = self.def_result.as_mut().expect("checked above");
        frame.value = returned;
        frame.returned = true;
        Ok(())
    }

    /// Whether a def call in the running statement reached `end` under `parse`,
    /// so nothing after the call in that statement may run. `parse_dynamic`
    /// leaves the rest to the halt block it emitted.
    fn unwinds_end(&self) -> bool {
        self.exit == Some(Exit::End) && self.dynamic.is_none()
    }

    /// The error that carries an `end` out of an expression; `execute_block` drops it.
    fn end_unwind() -> PrismError {
        parse_error(0, String::new())
    }

    pub(super) fn resolve_calls(
        &mut self,
        expr: &Expr<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<Expr<'a>> {
        with_stack(|| {
            Ok(match expr {
                Expr::Call(call) => {
                    let args = call
                        .args
                        .iter()
                        .map(|arg| self.resolve_calls(arg, line, out))
                        .collect::<Result<Vec<_>>>()?;
                    if self.def_defs.contains_key(call.name) {
                        let args = args
                            .into_iter()
                            .map(|arg| match arg {
                                Expr::Element(element) if element.indices.len() == 1 => {
                                    Argument::Operand(Operand {
                                        name: OperandName::Register(element.array),
                                        index: Some(Index::Single(element.indices[0].clone())),
                                        line,
                                    })
                                }
                                value => Argument::Value(value),
                            })
                            .collect::<Vec<_>>();
                        if self.streaming() {
                            self.flush(std::mem::take(out));
                        }
                        let (instructions, value) =
                            self.expand_def(call.name, &args, None, line)?;
                        out.extend(instructions);
                        if self.unwinds_end() {
                            return Err(Self::end_unwind());
                        }
                        if self.streaming() {
                            self.flush(std::mem::take(out));
                        }
                        value.ok_or_else(|| {
                            parse_error(
                                line,
                                format!(
                                    "def `{}` has no scalar result for this expression",
                                    call.name
                                ),
                            )
                        })?
                    } else {
                        Expr::Call(Box::new(syntax_expr::Call {
                            name: call.name,
                            args,
                        }))
                    }
                }
                Expr::Binary { op, left, right } => {
                    let left = self.resolve_calls(left, line, out)?;
                    if matches!(op, syntax_expr::BinaryOp::And | syntax_expr::BinaryOp::Or) {
                        match self.typed_of(&left, line) {
                            Ok(value) if value.truthy() == (*op == syntax_expr::BinaryOp::Or) => {
                                return Ok(Expr::Number(ClassicalValue::Bool(value.truthy())));
                            }
                            Err(_)
                                if right
                                    .finds_call(&|name| self.def_defs.contains_key(name))
                                    .is_some() =>
                            {
                                return self.resolve_short_circuit(*op, &left, right, line, out);
                            }
                            _ => {}
                        }
                    }
                    Expr::Binary {
                        op: *op,
                        left: Box::new(left),
                        right: Box::new(self.resolve_calls(right, line, out)?),
                    }
                }
                Expr::Negate(inner) => {
                    Expr::Negate(Box::new(self.resolve_calls(inner, line, out)?))
                }
                Expr::Not(inner) => Expr::Not(Box::new(self.resolve_calls(inner, line, out)?)),
                Expr::BitNot(inner) => {
                    Expr::BitNot(Box::new(self.resolve_calls(inner, line, out)?))
                }
                Expr::Cast(cast) => Expr::Cast(Box::new(syntax_expr::Cast {
                    ty: cast.ty,
                    width: cast.width.clone(),
                    value: self.resolve_calls(&cast.value, line, out)?,
                })),
                Expr::Element(element) => Expr::Element(Box::new(syntax_expr::Element {
                    array: element.array,
                    indices: element
                        .indices
                        .iter()
                        .map(|index| self.resolve_calls(index, line, out))
                        .collect::<Result<_>>()?,
                })),
                _ => expr.clone(),
            })
        })
    }

    fn resolve_arguments<'p>(
        &mut self,
        args: &'p [Argument<'a>],
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<Cow<'p, [Argument<'a>]>> {
        if self.def_defs.is_empty()
            || !args.iter().any(|arg| {
                matches!(
                    arg, Argument::Value(value)
                        if value.finds_call(&|name| self.def_defs.contains_key(name)).is_some()
                )
            })
        {
            return Ok(Cow::Borrowed(args));
        }
        args.iter()
            .map(|arg| {
                Ok(match arg {
                    Argument::Value(value) => {
                        Argument::Value(self.resolve_calls(value, line, out)?)
                    }
                    Argument::Operand(_) => arg.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Cow::Owned)
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
                    StmtKind::For { body, .. }
                    | StmtKind::While { body, .. }
                    | StmtKind::Box { body, .. } => {
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
    pub(super) fn pick_indices(
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
        self.reject_symbol(expr, line)?;
        let expr = self.with_complex(expr)?;
        let expr = self.fold_externs(&expr, line)?;
        let vars = self.param_vars.as_ref();
        if !self.classical.is_empty() && expr.reads_element() {
            let resolved = replace_nodes(&expr, &|node| {
                let Expr::Element(element) = node else {
                    return Ok(None);
                };
                let Some(bits) = self.int_bits(element, line)? else {
                    return Ok(None);
                };
                let value = vars.and_then(|vars| vars.get(element.array));
                Ok(value.map(|value| {
                    let word = value.to_int();
                    let packed = bits
                        .iter()
                        .enumerate()
                        .fold(0i128, |acc, (at, &bit)| acc | (((word >> bit) & 1) << at));
                    Expr::Number(ClassicalValue::Int(packed))
                }))
            })?;
            return number_of(&resolved, line, vars, &self.durations, &self.arrays);
        }
        number_of(&expr, line, vars, &self.durations, &self.arrays)
    }

    /// The bits `x[i]` or `x[a:b]` reads of an `int[n]` or `uint[n]` `x`, the
    /// lowest bit of the value it reads as first; `None` for any other element.
    fn int_bits(
        &self,
        element: &syntax_expr::Element,
        line: usize,
    ) -> Result<Option<SmallVec<[u32; 4]>>> {
        use crate::circuit::dynamic::ClassicalType as Ty;
        if let [Expr::Range(range)] = element.indices.as_slice()
            && !self.arrays.contains_key(element.array)
            && let Some(ClassicalDecl {
                ty: ClassicalType::Int(Ty::Int { width } | Ty::Uint { width }),
                ..
            }) = self.classical.get(element.array)
        {
            let picks = range_positions(range, *width as usize, element.array, line, |index| {
                self.typed_of(index, line)
            })?;
            return Ok(Some(picks.into_iter().map(|at| at as u32).collect()));
        }
        Ok(self.int_bit(element, line)?.map(|at| smallvec![at]))
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
        params: &[Argument<'a>],
        operands: &[Operand],
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let resolved = self.resolve_arguments(params, line, out)?;
        let params = resolved.as_ref();
        if self.externs.contains_key(name) {
            return self.exec_extern_statement(modifiers, name, params, operands, line);
        }
        if self.dynamic.is_some()
            && !self.def_defs.contains_key(name)
            && params.iter().any(|param| self.argument_is_runtime(param))
        {
            return self.exec_runtime_rotation(modifiers, name, params, operands, line, out);
        }
        let modifiers = self.fold_modifiers(modifiers, line)?;

        if self.def_defs.contains_key(name) {
            if !modifiers.is_empty() && self.def_is_runtime(name, &self.def_defs[name].args, params)
            {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("a modifier on runtime subroutine `{name}`"),
                    line,
                });
            }
            if self.streaming() {
                self.flush(std::mem::take(out));
            }
            let (instrs, _) = self.expand_def(name, params, None, line)?;
            if !modifiers.is_empty()
                && instrs
                    .iter()
                    .any(|instr| gate_angle(instr).and_then(tag_id).is_some())
            {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("modifier on `{name}` reading an `input`"),
                    line,
                });
            }
            out.extend(Self::modify_expansion(instrs, &modifiers, name, line)?);
            return Ok(());
        }

        let values = self.fold_params(params, line)?;
        let has_input = values.iter().any(|value| tag_id(*value).is_some());
        if has_input && (name == "gphase" || !modifiers.is_empty()) {
            let what = if name == "gphase" {
                "`gphase`".to_string()
            } else {
                format!("modifier on `{name}`")
            };
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("{what} reading an `input`"),
                line,
            });
        }
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

        let resolved: SmallVec<[SmallVec<[usize; 4]>; 4]> = operands
            .iter()
            .map(|operand| self.qubits_of(operand))
            .collect::<Result<_>>()?;
        let width = self.broadcast_length(&resolved, name, line)?;

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
                name, &values, &modifiers, &qubits, has_input, line, out,
            )?;
        }
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
        if !values.contains(&true)
            && !self
                .def_result
                .as_ref()
                .is_some_and(|frame| frame.runtime && frame.local == Some(name))
        {
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

    /// Evaluate a call's angle arguments. Two reading an `input` are rejected: a
    /// slot is written onto the single angle a bindable gate carries.
    fn fold_params(&self, params: &[Argument], line: usize) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(params.len());
        let mut tagged = false;
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
            let value = self.fold_angle(expr, line)?;
            if tag_id(value).is_some() {
                if tagged {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: "two `input` parameters on one gate".to_string(),
                        line,
                    });
                }
                tagged = true;
            }
            values.push(value);
        }
        Ok(values)
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
        let index = match &operand.index {
            Some(Index::Single(index)) => index.clone(),
            Some(Index::Range(range)) => Expr::Range(range.clone()),
            _ => return None,
        };
        Some(Expr::Element(Box::new(syntax_expr::Element {
            array,
            indices: vec![index],
        })))
    }

    /// One angle argument. One that reads an `input` comes back as a tag for its
    /// affine form, or as a number when that form is constant.
    fn fold_angle(&self, expr: &Expr, line: usize) -> Result<f64> {
        if self.symbol_in(expr).is_none() {
            return self.value_of(expr, line);
        }
        let form = self.affine_of(expr, line)?;
        Ok(match form.slot {
            Some(_) if form.scale != 0.0 => self.tag(form),
            _ => form.offset,
        })
    }

    /// `expr` as an affine form over the one slot its tagged names read.
    fn affine_of(&self, expr: &Expr, line: usize) -> Result<Linear> {
        if self.symbol_in(expr).is_none() {
            return Ok(Linear {
                slot: None,
                scale: 0.0,
                offset: self.value_of(expr, line)?,
            });
        }
        let scaled = |form: Linear, by: f64| Linear {
            scale: form.scale * by,
            offset: form.offset * by,
            ..form
        };
        with_stack(|| match expr {
            Expr::Ident(name) => Ok(self.symbol(name).expect("symbol_in found it")),
            Expr::Negate(inner) => Ok(scaled(self.affine_of(inner, line)?, -1.0)),
            Expr::Binary { op, left, right } => {
                let (left, right) = (self.affine_of(left, line)?, self.affine_of(right, line)?);
                let sign = match op {
                    syntax_expr::BinaryOp::Add => 1.0,
                    syntax_expr::BinaryOp::Sub => -1.0,
                    syntax_expr::BinaryOp::Mul if left.slot.is_none() => {
                        return Ok(scaled(right, left.offset));
                    }
                    syntax_expr::BinaryOp::Mul if right.slot.is_none() => {
                        return Ok(scaled(left, right.offset));
                    }
                    syntax_expr::BinaryOp::Div if right.slot.is_none() && right.offset != 0.0 => {
                        return Ok(Linear {
                            scale: left.scale / right.offset,
                            offset: left.offset / right.offset,
                            ..left
                        });
                    }
                    _ => return Err(non_affine(expr, line)),
                };
                if left.slot.is_some() && right.slot.is_some() && left.slot != right.slot {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("the angle `{expr}`, which reads two `input`s"),
                        line,
                    });
                }
                Ok(Linear {
                    slot: left.slot.or(right.slot),
                    scale: left.scale + sign * right.scale,
                    offset: left.offset + sign * right.offset,
                })
            }
            _ => Err(non_affine(expr, line)),
        })
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
        if call_params.iter().any(|value| tag_id(*value).is_some()) {
            sub.symbolic = def.params.clone();
        }
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

    /// Evaluate an `extern` call written as a statement and drop its value. A
    /// call over runtime values is dropped unevaluated, since nothing reads it.
    fn exec_extern_statement<'e>(
        &self,
        modifiers: &[ast::Modifier],
        name: &'e str,
        params: &[Argument<'e>],
        operands: &[Operand],
        line: usize,
    ) -> Result<()> {
        if !modifiers.is_empty() || !operands.is_empty() {
            return Err(parse_error(
                line,
                format!("extern `{name}` is a function, not a gate, and takes no qubits"),
            ));
        }
        let call = Expr::Call(Box::new(syntax_expr::Call {
            name,
            args: params
                .iter()
                .map(|param| match param {
                    Argument::Value(value) => Ok(value.clone()),
                    Argument::Operand(operand) => operand_reference(operand),
                })
                .collect::<Result<_>>()?,
        }));
        if !self.reads_runtime(&call) {
            self.typed_of(&call, line)?;
        }
        Ok(())
    }

    /// Assign a subroutine result or a builtin value to the caller's target.
    fn exec_call_assign(
        &mut self,
        assign: &ast::CallAssign<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let args = self.resolve_arguments(&assign.args, line, out)?;
        if self.def_defs.contains_key(assign.name) && self.streaming() {
            self.flush(std::mem::take(out));
        }
        if let Some(def) = self.def_defs.get(assign.name) {
            if def.result.as_ref().is_some_and(|result| result.ty != "bit") {
                let (instructions, value) = self.expand_def(assign.name, &args, None, line)?;
                out.extend(instructions);
                if self.unwinds_end() {
                    return Ok(());
                }
                if self.streaming() {
                    self.flush(std::mem::take(out));
                }
                let value = value.expect("typed result");
                return self.assign_call_result(&assign.target, value, line);
            }
            let bits = self.bits_of(&assign.target)?;
            let (instructions, _) = self.expand_def(assign.name, &args, Some(&bits), line)?;
            out.extend(instructions);
            return Ok(());
        }
        if self.externs.contains_key(assign.name) {
            let value = Expr::Call(Box::new(syntax_expr::Call {
                name: assign.name,
                args: args
                    .iter()
                    .map(|arg| match arg {
                        Argument::Value(value) => Ok(value.clone()),
                        Argument::Operand(operand) => operand_reference(operand),
                    })
                    .collect::<Result<_>>()?,
            }));
            return self.assign_call_result(&assign.target, value, line);
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
        let args = args
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

    /// Write a call's scalar result to a variable, an array element or bits.
    fn assign_call_result(
        &mut self,
        target: &Operand<'a>,
        value: Expr<'a>,
        line: usize,
    ) -> Result<()> {
        let name = target
            .register()
            .ok_or_else(|| parse_error(line, "a scalar result needs a classical target"))?;
        if self.is_bit_name(name) {
            return self.assign_bits(target, None, &value, line);
        }
        match &target.index {
            None => self.assign_classical(name, None, &value, line),
            Some(Index::Single(index)) => self.assign_element(
                &ast::ElementAssign {
                    array: name,
                    indices: vec![index.clone()],
                    op: None,
                    value,
                },
                line,
            ),
            _ => Err(parse_error(line, "a scalar result needs one array element")),
        }
    }

    /// Inline a `def` call, binding each argument by the kind its declaration
    /// gave it and the result, when there is one, to `result`.
    fn expand_def(
        &mut self,
        name: &str,
        args: &[Argument],
        result: Option<&[usize]>,
        line: usize,
    ) -> Result<(Vec<Instruction>, Option<Expr<'a>>)> {
        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(parse_error(
                line,
                format!(
                    "def expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive call to `{name}`"
                ),
            ));
        }
        let definitions = Rc::clone(&self.def_defs);
        let def = definitions.get(name).expect("checked by the caller");
        if args.len() != def.args.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.args.len(),
                got: args.len(),
            });
        }

        let copied = 2 * def.constants.len() + def.arrays.len() + args.len();
        self.charge(copied, line)?;
        let mut bindings = HashMap::new();
        let mut symbolic = Vec::new();
        let runtime = self.def_is_runtime(name, &def.args, args);
        let mut scalar_args = Vec::new();
        let mut durations = HashMap::new();
        let mut complex = HashMap::new();
        let mut array_args = Vec::new();
        for (&name, decl) in def.constants.iter() {
            match decl.value {
                Some(ConstValue::Number(value)) => {
                    bindings.insert(name, value);
                }
                Some(ConstValue::Complex(value)) => {
                    complex.insert(name, value);
                }
                Some(ConstValue::Duration(value)) => {
                    durations.insert(name, value);
                }
                None => {}
            }
        }
        let mut qubit_bindings: Vec<(&'a str, SmallVec<[usize; 4]>)> = Vec::new();
        let mut bound_qubits = HashSet::new();
        let mut bit_bindings: Vec<(&'a str, SmallVec<[usize; 4]>)> = Vec::new();
        for (slot, argument) in def.args.iter().zip(args) {
            match slot {
                DefParam::Qubit {
                    name: param_name,
                    width,
                } => {
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
                    let qubits = self.qubits_of(&operand)?;
                    self.charge(qubits.len(), line)?;
                    let wanted = match width {
                        Some(width) => self.integer_of(width, line)?,
                        None => 1,
                    };
                    if wanted <= 0 || qubits.len() as i64 != wanted {
                        return Err(parse_error(
                            line,
                            format!(
                                "def `{name}` qubit parameter `{param_name}` takes {wanted} qubit(s), got {}",
                                qubits.len()
                            ),
                        ));
                    }
                    if qubits.iter().any(|&qubit| !bound_qubits.insert(qubit)) {
                        return Err(parse_error(
                            line,
                            format!("def `{name}` passes a qubit more than once"),
                        ));
                    }
                    qubit_bindings.push((param_name, qubits));
                }
                DefParam::Value {
                    name: param_name,
                    ty,
                    width,
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
                    let kind = self.classical_type(ty, width.as_ref(), line)?;
                    // An `input` passes through as its tag, which the body reads
                    // only as a gate angle. An angle parameter keeps it unwrapped,
                    // as a bound `input angle` is.
                    if let Some(input) = self.symbol_in(expr) {
                        if !matches!(
                            kind,
                            ClassicalType::Float(64 | 128) | ClassicalType::Angle(None)
                        ) {
                            return Err(self.symbol_error(
                                input,
                                &format!(
                                    "as `def {name}` parameter `{param_name}`, whose type rounds it"
                                ),
                                line,
                            ));
                        }
                        bindings.insert(
                            param_name,
                            ClassicalValue::Float(self.fold_angle(expr, line)?),
                        );
                        symbolic.push(param_name);
                        continue;
                    }
                    if let ClassicalType::Complex(_) = kind {
                        if runtime {
                            return Err(super::dynamic::runtime_complex(line));
                        }
                        complex.insert(param_name, self.complex_of(expr, line)?);
                        continue;
                    }
                    if runtime {
                        let lowered = match self.lower_expr(expr, line)? {
                            ClassicalExpr::Const(_) => {
                                ClassicalExpr::Const(self.fold_typed(kind, expr, line)?)
                            }
                            value => value,
                        };
                        scalar_args.push((param_name, kind, lowered));
                    } else {
                        bindings.insert(param_name, self.fold_typed(kind, expr, line)?);
                    }
                }
                DefParam::Duration(param_name) => {
                    let Argument::Value(expr) = argument else {
                        return Err(parse_error(
                            line,
                            format!("def `{name}` parameter `{param_name}` takes a duration"),
                        ));
                    };
                    let value = timed_of(
                        expr,
                        line,
                        self.param_vars.as_ref(),
                        &self.durations,
                        &self.arrays,
                    )?;
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
                    if runtime
                        && let Some((size, value)) = self.runtime_bit_argument(argument, line)?
                    {
                        let wanted = width
                            .as_ref()
                            .map(|width| self.integer_of(width, line))
                            .transpose()?
                            .unwrap_or(1);
                        if size as i64 != wanted {
                            return Err(parse_error(
                                line,
                                format!(
                                    "def `{name}` bit parameter `{param_name}` takes {wanted} bit(s), got {size}"
                                ),
                            ));
                        }
                        scalar_args.push((
                            *param_name,
                            ClassicalType::Int(RuntimeType::Uint { width: size as u32 }),
                            value,
                        ));
                        continue;
                    }
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
                    if runtime {
                        let value = super::dynamic::bits_value(&bits, line)?;
                        let kind = RuntimeType::Uint {
                            width: bits.len() as u32,
                        };
                        scalar_args.push((*param_name, ClassicalType::Int(kind), value));
                    } else {
                        bit_bindings.push((param_name, bits));
                    }
                }
                DefParam::Array {
                    name: param_name,
                    mutable,
                    ty,
                    width,
                    shape,
                } => {
                    let reference = match argument {
                        Argument::Value(expr) => expr.clone(),
                        Argument::Operand(operand) => operand_reference(operand)?,
                    };
                    let (source, indices) = match &reference {
                        Expr::Ident(source) => (*source, &[][..]),
                        Expr::Element(element) => (element.array, element.indices.as_slice()),
                        _ => ("", &[][..]),
                    };
                    let Some(array) = self.arrays.get(source) else {
                        return Err(parse_error(
                            line,
                            format!(
                                "def `{name}` parameter `{param_name}` takes an array, got \
                                 `{reference}`"
                            ),
                        ));
                    };
                    let kind = self.array_kind(ty, width.as_ref(), line)?;
                    if kind != array.ty {
                        return Err(parse_error(
                            line,
                            format!(
                                "def `{name}` parameter `{param_name}` takes `{ty}` elements, \
                                 which `{source}` does not hold"
                            ),
                        ));
                    }
                    let view = array_view(array, source, indices, line, |index| {
                        self.typed_of(index, line)
                    })?;
                    let fits = match shape {
                        ArrayShape::Sized(dims) => {
                            let wanted = dims
                                .iter()
                                .map(|size| self.integer_of(size, line))
                                .collect::<Result<Vec<_>>>()?;
                            wanted.len() == view.dims.len()
                                && wanted
                                    .iter()
                                    .zip(&view.dims)
                                    .all(|(&wanted, &size)| wanted == size as i64)
                        }
                        ArrayShape::Rank(rank) => {
                            self.integer_of(rank, line)? == view.dims.len() as i64
                        }
                    };
                    if !fits {
                        return Err(parse_error(
                            line,
                            format!(
                                "def `{name}` parameter `{param_name}` cannot take `{reference}`, \
                                 whose shape is {:?}",
                                view.dims
                            ),
                        ));
                    }
                    if *mutable && array.constant {
                        return Err(parse_error(
                            line,
                            format!("`{source}` is `const` and cannot pass as a `mutable` array"),
                        ));
                    }
                    if *mutable && runtime {
                        return Err(PrismError::UnsupportedConstruct {
                            construct: format!(
                                "a `mutable` array argument to def `{name}`, whose call runs \
                                 at runtime"
                            ),
                            line,
                        });
                    }
                    self.charge(view.offsets.len(), line)?;
                    let bound = ClassicalArray {
                        ty: kind,
                        constant: !*mutable,
                        dims: view.dims,
                        values: view.offsets.iter().map(|&at| array.values[at]).collect(),
                    };
                    let write_back = mutable.then(|| (source.to_string(), view.offsets));
                    array_args.push((*param_name, Rc::new(bound), write_back));
                }
            }
        }

        let frame = match (&def.result, result) {
            (None, None) => DefResult {
                bits: SmallVec::new(),
                local: None,
                ty: None,
                value: None,
                returned: false,
                runtime: false,
                result_var: None,
            },
            (None, Some(_)) => {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares no result to assign"),
                ));
            }
            (Some(declared), None) if declared.ty != "bit" => DefResult {
                bits: SmallVec::new(),
                local: None,
                ty: Some(self.classical_type(declared.ty, declared.width.as_ref(), line)?),
                value: None,
                returned: false,
                runtime: false,
                result_var: None,
            },
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
                    ty: None,
                    value: None,
                    returned: false,
                    runtime: false,
                    result_var: None,
                }
            }
        };

        let mut sub = self.expansion_parser(bindings);
        sub.symbolic = symbolic;
        // The body runs in a parser of its own, so a write it made to an outer
        // variable would be dropped on return; only the constants it may read
        // stay declared, and a write to anything else is rejected.
        sub.classical = Rc::new(
            def.constants
                .iter()
                .filter(|(_, decl)| decl.constant)
                .map(|(name, decl)| (*name, decl.clone()))
                .collect(),
        );
        sub.durations = durations;
        sub.complex = complex;
        // A def sees the global constants and nothing else it does not take as
        // an argument, so only a `const` array reaches its body.
        sub.arrays = def.arrays.clone();
        let mut write_backs = Vec::new();
        for (param_name, array, write_back) in array_args {
            sub.arrays.insert(param_name, array);
            if let Some(write_back) = write_back {
                write_backs.push((param_name, write_back));
            }
        }
        sub.qregs = Rc::new(HashMap::new());
        sub.cregs = Rc::new(HashMap::new());
        sub.aliases = Rc::new(HashMap::new());
        for (param_name, qubits) in qubit_bindings {
            sub.params.insert(
                param_name,
                Alias {
                    kind: ast::RegisterKind::Qubit,
                    indices: qubits.into_vec(),
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
        if runtime {
            let value = self.execute_runtime_def(name, def, sub, scalar_args, line)?;
            return Ok((Vec::new(), value));
        }
        for param in &def.args {
            if let DefParam::Value { name, ty, width } = param {
                let kind = sub.classical_type(ty, width.as_ref(), line)?;
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
            }
        }
        let instructions = sub.execute(&def.body)?;
        for (param_name, (source, offsets)) in write_backs {
            if let (Some(written), Some(array)) = (
                sub.arrays.get(param_name),
                self.arrays.get_mut(source.as_str()),
            ) {
                let array = Rc::make_mut(array);
                for (&at, &value) in offsets.iter().zip(&written.values) {
                    array.values[at] = value;
                }
            }
        }
        let frame = sub.def_result.expect("installed above");
        if sub.exit == Some(Exit::End) {
            self.exec_end(line)?;
            let value = frame.value.or_else(|| {
                def.result
                    .is_some()
                    .then_some(Expr::Number(ClassicalValue::Int(0)))
            });
            return Ok((instructions, value));
        }
        if def.local.is_some() {
            self.charge(frame.bits.len(), line)?;
            let mut initialized = frame.bits.iter().map(|&bit| (bit, false)).collect();
            if self.returned_bits_need_initialization(&instructions, &mut initialized, line)?
                || initialized.values().any(|initialized| !initialized)
            {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "an initializer on the bit returned by def `{name}`; use `parse_dynamic`"
                    ),
                    line,
                });
            }
        }
        if def.result.is_some() && !frame.returned {
            return Err(parse_error(
                line,
                format!("def `{name}` reaches its end without a result"),
            ));
        }
        Ok((instructions, frame.value))
    }

    fn returned_bits_need_initialization(
        &self,
        body: &[Instruction],
        initialized: &mut HashMap<usize, bool>,
        line: usize,
    ) -> Result<bool> {
        with_stack(|| {
            let mut read = Vec::new();
            for instruction in body {
                let condition = match instruction {
                    Instruction::Measure { classical_bit, .. } => {
                        if let Some(bit) = initialized.get_mut(classical_bit) {
                            *bit = true;
                        }
                        continue;
                    }
                    Instruction::Conditional { condition, .. } => Some(condition),
                    Instruction::Region(region) => Some(region.condition()),
                    _ => None,
                };
                if let Some(condition) = condition {
                    read.clear();
                    crate::circuit::collect_condition_bits(condition, &mut read);
                    self.charge(read.len(), line)?;
                    if read.iter().any(|bit| initialized.get(bit) == Some(&false)) {
                        return Ok(true);
                    }
                }
                if let Instruction::Region(region) = instruction {
                    self.charge(initialized.len(), line)?;
                    if self.returned_bits_need_initialization(
                        region.body(),
                        &mut initialized.clone(),
                        line,
                    )? {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        })
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
            externs: Rc::clone(&self.externs),
            registry: self.registry.clone(),
            total_qubits: self.total_qubits,
            total_cbits: self.total_cbits,
            gate_expansion_depth: self.gate_expansion_depth + 1,
            region_depth: self.region_depth,
            exec_depth: self.exec_depth,
            param_vars: Some(bindings),
            inputs: HashMap::new(),
            input_names: Vec::new(),
            tags: Rc::clone(&self.tags),
            symbolic: Vec::new(),
            supplied: Vec::new(),
            nested: true,
            results: Vec::new(),
            noise_specs: Vec::new(),
            pending_noise: None,
            verbatim_pending: false,
            physical: self.physical,
            aliases: Rc::clone(&self.aliases),
            params: HashMap::new(),
            classical: Rc::clone(&self.classical),
            durations: HashMap::new(),
            complex: HashMap::new(),
            arrays: HashMap::new(),
            declared: Vec::new(),
            def_result: None,
            loops: Vec::new(),
            exit: None,
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
        out: &mut Vec<Instruction>,
    ) -> Result<Vec<Instruction>> {
        if let Ok(expr) = super::dynamic::condition_expr(&conditional.condition)
            && expr
                .finds_call(&|name| self.def_defs.contains_key(name))
                .is_some()
        {
            let value = self.resolve_calls(&expr, line, out)?;
            let ClassicalExpr::Const(value) = self.lower_expr(&value, line)? else {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "a def call in a measurement-dependent condition".to_string(),
                    line,
                });
            };
            let body = if value.truthy() {
                Some(&conditional.then_body)
            } else {
                conditional.else_body.as_ref()
            };
            return body.map_or(Ok(Vec::new()), |body| self.exec_box(body));
        }

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
            self.complex.remove(name);
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
        let body = self.execute(block).and_then(|body| {
            if body
                .iter()
                .any(|instr| gate_angle(instr).and_then(tag_id).is_some())
            {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "an angle reading an `input` inside a guarded region".to_string(),
                    line: block.first().map_or(1, |stmt| stmt.line),
                });
            }
            Ok(body)
        });
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
        stmt: &Stmt<'a>,
        variable: &'a str,
        range: &ForRange<'a>,
        body: &Block<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
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
        // The loop variable may shadow an outer name, whose value comes back
        // once the loop ends.
        let shadowed = self
            .param_vars
            .as_ref()
            .and_then(|vars| vars.get(variable).copied());
        let runtime_binding = self
            .dynamic
            .as_mut()
            .and_then(|state| state.vars.remove(variable));
        let bit_binding = self
            .dynamic
            .as_mut()
            .and_then(|state| state.bit_values.remove(variable));
        let shadowed_decl = Rc::make_mut(&mut self.classical).insert(
            variable,
            ClassicalDecl {
                ty: ClassicalType::Int(RuntimeType::Int { width: 64 }),
                constant: false,
                value: None,
            },
        );
        let scope = self.open_scope();
        let mut result = Ok(());
        self.enter_static_loop();
        let graph = self.unrolls_into_graph(stmt);
        if graph {
            self.flush(std::mem::take(out));
            self.begin_unrolled();
        }
        self.loops.push(Loop::For {
            region_depth: self.region_depth,
            frames: self.graph_depth(),
            graph,
        });
        for (pass, value) in values.into_iter().enumerate() {
            if let Err(error) = self.charge(1, line) {
                result = Err(error);
                break;
            }
            let mark = if graph { self.pass_mark() } else { (0, 0) };
            if graph
                && pass > 0
                && let Err(error) = self.next_pass()
            {
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
                Ok(produced) => {
                    if graph {
                        self.flush(produced);
                        if let Err(error) = self.charge_pass_blocks(mark, line) {
                            result = Err(error);
                            break;
                        }
                    } else {
                        out.extend(produced);
                    }
                    // A path a runtime `continue` sent to the next pass still
                    // needs that pass, so a parse-time exit after one jumps
                    // instead of ending the unrolling.
                    let continued = graph && self.pass_continued();
                    match self.exit {
                        Some(Exit::Break) => {
                            self.exit = None;
                            if !continued {
                                break;
                            }
                            if let Err(error) = self.break_unrolled() {
                                result = Err(error);
                                break;
                            }
                        }
                        Some(Exit::Continue) => self.exit = None,
                        Some(Exit::End) => break,
                        None => {}
                    }
                    if let Some(frame) = self.def_result.as_mut().filter(|frame| frame.returned) {
                        if !continued {
                            break;
                        }
                        frame.returned = false;
                    }
                }
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.loops.pop();
        if graph && result.is_ok() {
            // Paths a runtime `break` sent to the exit go on past the loop,
            // whatever `return` the rest took.
            if self.loop_broke()
                && let Some(frame) = self.def_result.as_mut()
            {
                frame.returned = false;
            }
            result = self.end_unrolled();
        }
        self.leave_static_loop();
        if let Some(state) = self.dynamic.as_mut() {
            if let Some(var) = runtime_binding {
                state.vars.insert(variable, var);
            }
            if let Some(width) = bit_binding {
                state.bit_values.insert(variable, width);
            }
        }
        match shadowed_decl {
            Some(decl) => {
                Rc::make_mut(&mut self.classical).insert(variable, decl);
            }
            None => {
                Rc::make_mut(&mut self.classical).remove(variable);
            }
        }
        if let Some(vars) = self.param_vars.as_mut() {
            match shadowed {
                Some(value) => vars.insert(variable, value),
                None => vars.remove(variable),
            };
        }
        result
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
