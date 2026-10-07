//! Expression syntax tree, its parser, and its evaluator.
//!
//! Parsing and evaluation are separate so a body that runs many times, a `for`
//! body above all, is parsed once and evaluated per pass.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use super::lexer::{Kind, TIME_UNITS};
use super::stream::Stream;
use crate::error::{PrismError, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

impl BinaryOp {
    fn spelling(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
            BinaryOp::Pow => "**",
        }
    }
}

/// A length of time, held as nanoseconds beside backend sample periods (`dt`),
/// which only a backend relates to each other.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub(crate) struct Duration {
    pub ns: f64,
    pub dt: f64,
}

impl Duration {
    fn scaled(self, factor: f64) -> Duration {
        Duration {
            ns: self.ns * factor,
            dt: self.dt * factor,
        }
    }

    pub(crate) fn is_negative(self) -> bool {
        self.ns < 0.0 || self.dt < 0.0
    }
}

/// What a timing expression folds to. A stretch has no length until a scheduler
/// sizes it, so anything built from one stays unresolved.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Timed {
    Number(f64),
    Duration(Duration),
    Stretch,
}

impl Timed {
    fn describe(self) -> &'static str {
        match self {
            Timed::Number(_) => "a number",
            Timed::Duration(_) => "a duration",
            Timed::Stretch => "a stretch",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Expr<'a> {
    Number(f64),
    Duration(Duration),
    Ident(&'a str),
    Negate(Box<Expr<'a>>),
    Binary {
        op: BinaryOp,
        left: Box<Expr<'a>>,
        right: Box<Expr<'a>>,
    },
    /// Boxed so the one wide variant does not set the width of every node.
    Call(Box<Call<'a>>),
    /// An array element, `a[i, j]` or `a[i][j]`.
    Element(Box<Element<'a>>),
}

#[derive(Clone, Debug)]
pub(crate) struct Call<'a> {
    pub name: &'a str,
    pub args: Vec<Expr<'a>>,
}

#[derive(Clone, Debug)]
pub(crate) struct Element<'a> {
    pub array: &'a str,
    pub indices: Vec<Expr<'a>>,
}

impl<'a> Expr<'a> {
    /// True when the expression reads `name`, which decides whether it depends
    /// on an `input` slot.
    pub(crate) fn mentions(&self, name: &str) -> bool {
        match self {
            Expr::Number(_) | Expr::Duration(_) => false,
            Expr::Ident(ident) => *ident == name,
            Expr::Negate(inner) => inner.mentions(name),
            Expr::Binary { left, right, .. } => left.mentions(name) || right.mentions(name),
            Expr::Call(call) => call.args.iter().any(|arg| arg.mentions(name)),
            Expr::Element(element) => element.indices.iter().any(|index| index.mentions(name)),
        }
    }

    /// True when the expression reads an array element.
    pub(crate) fn reads_element(&self) -> bool {
        match self {
            Expr::Number(_) | Expr::Duration(_) | Expr::Ident(_) => false,
            Expr::Negate(inner) => inner.reads_element(),
            Expr::Binary { left, right, .. } => left.reads_element() || right.reads_element(),
            Expr::Call(call) => call.args.iter().any(Expr::reads_element),
            Expr::Element(_) => true,
        }
    }

    /// The identifier this expression is, when it is exactly one and nothing
    /// more. An `input` binds an angle whole, so that is the shape it takes.
    pub(crate) fn as_ident(&self) -> Option<&'a str> {
        match self {
            Expr::Ident(name) => Some(name),
            _ => None,
        }
    }
}

impl fmt::Display for Expr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(value) => {
                // A whole number reads back as it was written rather than as a
                // float, since that is how an index or a bound is spelled.
                if value.fract() == 0.0 && value.abs() < 1e15 {
                    write!(f, "{}", *value as i64)
                } else {
                    write!(f, "{value}")
                }
            }
            Expr::Duration(duration) if duration.dt == 0.0 => write!(f, "{}ns", duration.ns),
            Expr::Duration(duration) => write!(f, "{}dt", duration.dt),
            Expr::Ident(name) => f.write_str(name),
            Expr::Negate(inner) => {
                f.write_str("-")?;
                grouped(f, inner)
            }
            Expr::Binary { op, left, right } => {
                grouped(f, left)?;
                write!(f, " {} ", op.spelling())?;
                grouped(f, right)
            }
            Expr::Call(call) => {
                write!(f, "{}(", call.name)?;
                for (at, arg) in call.args.iter().enumerate() {
                    if at > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                f.write_str(")")
            }
            Expr::Element(element) => {
                write!(f, "{}[", element.array)?;
                for (at, index) in element.indices.iter().enumerate() {
                    if at > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{index}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// A subexpression, parenthesised when it is itself an operator application so
/// the rendering reads back with the grouping it was parsed with.
fn grouped(f: &mut fmt::Formatter<'_>, expr: &Expr<'_>) -> fmt::Result {
    match expr {
        Expr::Binary { .. } => write!(f, "({expr})"),
        _ => write!(f, "{expr}"),
    }
}

/// Parse one expression, leaving the cursor on whatever follows it.
pub(crate) fn parse<'a>(stream: &mut Stream<'_, 'a>) -> Result<Expr<'a>> {
    parse_sum(stream)
}

fn parse_sum<'a>(stream: &mut Stream<'_, 'a>) -> Result<Expr<'a>> {
    let mut left = parse_product(stream)?;
    loop {
        let op = match stream.kind() {
            Kind::Plus => BinaryOp::Add,
            Kind::Minus => BinaryOp::Sub,
            _ => return Ok(left),
        };
        stream.advance();
        left = binary(op, left, parse_product(stream)?);
    }
}

fn parse_product<'a>(stream: &mut Stream<'_, 'a>) -> Result<Expr<'a>> {
    let mut left = parse_unary(stream)?;
    loop {
        let op = match stream.kind() {
            Kind::Star => BinaryOp::Mul,
            Kind::Slash => BinaryOp::Div,
            Kind::Percent => BinaryOp::Rem,
            _ => return Ok(left),
        };
        stream.advance();
        left = binary(op, left, parse_unary(stream)?);
    }
}

fn parse_unary<'a>(stream: &mut Stream<'_, 'a>) -> Result<Expr<'a>> {
    if stream.eat(Kind::Minus) {
        return Ok(Expr::Negate(Box::new(parse_unary(stream)?)));
    }
    if stream.eat(Kind::Plus) {
        return parse_unary(stream);
    }
    parse_power(stream)
}

/// `**` is right associative and binds tighter than unary minus, so `-2 ** 2`
/// is `-4` and `2 ** -1` is a power of a negation.
fn parse_power<'a>(stream: &mut Stream<'_, 'a>) -> Result<Expr<'a>> {
    let base = parse_primary(stream)?;
    if stream.eat(Kind::Pow) {
        return Ok(binary(BinaryOp::Pow, base, parse_unary(stream)?));
    }
    Ok(base)
}

fn parse_primary<'a>(stream: &mut Stream<'_, 'a>) -> Result<Expr<'a>> {
    match stream.kind() {
        Kind::LParen => {
            stream.advance();
            let inner = parse_sum(stream)?;
            if !stream.eat(Kind::RParen) {
                return Err(PrismError::Parse {
                    line: stream.line(),
                    message: "unmatched `(` in expression".into(),
                });
            }
            Ok(inner)
        }
        Kind::Int | Kind::Float => {
            let token = stream.advance();
            Ok(Expr::Number(number(token.text, token.line as usize)?))
        }
        Kind::Duration => {
            let token = stream.advance();
            Ok(Expr::Duration(duration(token.text, token.line as usize)?))
        }
        Kind::Ident => {
            let name = stream.advance().text;
            if stream.kind() == Kind::LBracket {
                return Ok(Expr::Element(Box::new(Element {
                    array: name,
                    indices: index_groups(stream)?,
                })));
            }
            if !stream.eat(Kind::LParen) {
                return Ok(Expr::Ident(name));
            }
            if name == "durationof" {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "`durationof`, which needs a scheduler to size the block".into(),
                    line: stream.line(),
                });
            }
            let mut args = vec![parse_sum(stream)?];
            while stream.eat(Kind::Comma) {
                args.push(parse_sum(stream)?);
            }
            if !stream.eat(Kind::RParen) {
                return Err(PrismError::Parse {
                    line: stream.line(),
                    message: format!("unmatched `(` after function `{name}`"),
                });
            }
            Ok(Expr::Call(Box::new(Call { name, args })))
        }
        Kind::Eof => Err(PrismError::Parse {
            line: stream.line(),
            message: "unexpected end of expression".into(),
        }),
        _ => Err(stream.expected("a value")),
    }
}

/// `[i, j]` or `[i][j]`, read as one flat index list.
pub(crate) fn index_groups<'a>(stream: &mut Stream<'_, 'a>) -> Result<Vec<Expr<'a>>> {
    let mut indices = Vec::new();
    while stream.eat(Kind::LBracket) {
        indices.push(parse_sum(stream)?);
        while stream.eat(Kind::Comma) {
            indices.push(parse_sum(stream)?);
        }
        stream.expect(Kind::RBracket)?;
    }
    Ok(indices)
}

fn binary<'a>(op: BinaryOp, left: Expr<'a>, right: Expr<'a>) -> Expr<'a> {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Read a numeric literal, in whichever radix it was written and with `_`
/// separators dropped.
fn number(text: &str, line: usize) -> Result<f64> {
    // Qubit indices are nearly every literal a program writes; fifteen digits
    // stay exact in an f64.
    if text.len() <= 15 && text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Ok(text
            .bytes()
            .fold(0u64, |acc, byte| acc * 10 + u64::from(byte - b'0')) as f64);
    }
    let invalid = || PrismError::Parse {
        line,
        message: format!("invalid number: `{text}`"),
    };
    let radix = match text.get(..2) {
        Some("0x" | "0X") => Some(16u32),
        Some("0b" | "0B") => Some(2),
        Some("0o" | "0O") => Some(8),
        _ => None,
    };
    let cleaned = if text.contains('_') {
        Cow::Owned(text.replace('_', ""))
    } else {
        Cow::Borrowed(text)
    };
    let value = match radix {
        Some(radix) => u64::from_str_radix(&cleaned[2..], radix).map_err(|_| invalid())? as f64,
        None => cleaned.parse::<f64>().map_err(|_| invalid())?,
    };
    if !value.is_finite() {
        return Err(PrismError::Parse {
            line,
            message: format!("value is not finite: `{text}`"),
        });
    }
    Ok(value)
}

/// Read a duration literal, its unit attached.
fn duration(text: &str, line: usize) -> Result<Duration> {
    let Some(unit) = TIME_UNITS.iter().find(|unit| text.ends_with(*unit)) else {
        return Err(PrismError::Parse {
            line,
            message: format!("invalid duration: `{text}`"),
        });
    };
    let value = number(&text[..text.len() - unit.len()], line)?;
    let nanoseconds = match *unit {
        "dt" => return Ok(Duration { ns: 0.0, dt: value }),
        "ns" => 1.0,
        "ms" => 1e6,
        "s" => 1e9,
        _ => 1e3,
    };
    Ok(Duration {
        ns: value * nanoseconds,
        dt: 0.0,
    })
}

/// Fold an expression to its value, resolving names against `vars`.
///
/// The finiteness check sits here rather than on each operator so that an
/// overflow built from finite parts is caught too.
pub(crate) fn eval(expr: &Expr, line: usize, vars: Option<&HashMap<&str, f64>>) -> Result<f64> {
    let value = evaluate(expr, line, vars)?;
    if !value.is_finite() {
        return Err(PrismError::Parse {
            line,
            message: format!(
                "the expression evaluates to {value} (must be finite); this typically                  means a divide by zero, a log or sqrt of a non-positive value, or an                  overflow"
            ),
        });
    }
    Ok(value)
}

fn evaluate(expr: &Expr, line: usize, vars: Option<&HashMap<&str, f64>>) -> Result<f64> {
    match expr {
        Expr::Number(value) => Ok(*value),
        Expr::Duration(_) => Err(PrismError::Parse {
            line,
            message: format!("`{expr}` is a duration where a number belongs"),
        }),
        Expr::Ident(name) => resolve(name, line, vars),
        Expr::Element(element) => Err(PrismError::Parse {
            line,
            message: format!("`{}` is not a declared array", element.array),
        }),
        Expr::Negate(inner) => Ok(-evaluate(inner, line, vars)?),
        Expr::Binary { op, left, right } => {
            let left = evaluate(left, line, vars)?;
            let right = evaluate(right, line, vars)?;
            arithmetic(*op, left, right, line)
        }
        Expr::Call(call) => {
            let values = call
                .args
                .iter()
                .map(|arg| evaluate(arg, line, vars))
                .collect::<Result<Vec<_>>>()?;
            apply(call.name, &values, line)
        }
    }
}

fn arithmetic(op: BinaryOp, left: f64, right: f64, line: usize) -> Result<f64> {
    match op {
        BinaryOp::Add => Ok(left + right),
        BinaryOp::Sub => Ok(left - right),
        BinaryOp::Mul => Ok(left * right),
        BinaryOp::Div => divide("division", left, right, line),
        BinaryOp::Rem => divide("modulo", left, right, line),
        BinaryOp::Pow => finite(line, op.spelling(), left.powf(right)),
    }
}

/// True when the expression reads a duration: a literal, or a name `times` holds.
pub(crate) fn is_timed(expr: &Expr, times: &HashMap<&str, Timed>) -> bool {
    match expr {
        Expr::Number(_) => false,
        Expr::Duration(_) => true,
        Expr::Ident(name) => times.contains_key(name),
        Expr::Negate(inner) => is_timed(inner, times),
        Expr::Binary { left, right, .. } => is_timed(left, times) || is_timed(right, times),
        Expr::Call(call) => call.args.iter().any(|arg| is_timed(arg, times)),
        Expr::Element(element) => element.indices.iter().any(|index| is_timed(index, times)),
    }
}

/// Fold an expression that may read durations, resolving the duration and
/// stretch names against `times` and every other name against `vars`.
///
/// Durations add and subtract, scale by a number, and divide into a number when
/// both sides are in SI units or both in `dt`. Any other mix is an error.
pub(crate) fn eval_timed(
    expr: &Expr,
    line: usize,
    vars: Option<&HashMap<&str, f64>>,
    times: &HashMap<&str, Timed>,
) -> Result<Timed> {
    if !is_timed(expr, times) {
        return Ok(Timed::Number(eval(expr, line, vars)?));
    }
    match expr {
        Expr::Duration(duration) => Ok(Timed::Duration(*duration)),
        Expr::Ident(name) => match times.get(name) {
            Some(value) => Ok(*value),
            None => Ok(Timed::Number(resolve(name, line, vars)?)),
        },
        Expr::Negate(inner) => Ok(match eval_timed(inner, line, vars, times)? {
            Timed::Number(value) => Timed::Number(-value),
            Timed::Duration(duration) => Timed::Duration(duration.scaled(-1.0)),
            Timed::Stretch => Timed::Stretch,
        }),
        Expr::Binary { op, left, right } => {
            let left = eval_timed(left, line, vars, times)?;
            let right = eval_timed(right, line, vars, times)?;
            combine_timed(*op, left, right, line)
        }
        Expr::Call(call) => Err(PrismError::Parse {
            line,
            message: format!("`{}` takes numbers, not durations", call.name),
        }),
        Expr::Element(element) => Err(PrismError::Parse {
            line,
            message: format!("`{expr}` indexes `{}` with a duration", element.array),
        }),
        Expr::Number(value) => Ok(Timed::Number(*value)),
    }
}

/// Apply `op` to two folded timing values.
pub(crate) fn combine_timed(op: BinaryOp, left: Timed, right: Timed, line: usize) -> Result<Timed> {
    use BinaryOp::{Add, Div, Mul, Sub};
    let zero_divisor = || PrismError::Parse {
        line,
        message: "division by zero in a duration expression".into(),
    };
    Ok(match (op, left, right) {
        (_, Timed::Number(a), Timed::Number(b)) => Timed::Number(arithmetic(op, a, b, line)?),
        (Add, Timed::Duration(a), Timed::Duration(b)) => Timed::Duration(Duration {
            ns: a.ns + b.ns,
            dt: a.dt + b.dt,
        }),
        (Sub, Timed::Duration(a), Timed::Duration(b)) => Timed::Duration(Duration {
            ns: a.ns - b.ns,
            dt: a.dt - b.dt,
        }),
        (Add | Sub, Timed::Stretch, Timed::Duration(_) | Timed::Stretch)
        | (Add | Sub, Timed::Duration(_), Timed::Stretch)
        | (Mul, Timed::Stretch, Timed::Number(_))
        | (Mul, Timed::Number(_), Timed::Stretch) => Timed::Stretch,
        (Mul, Timed::Duration(a), Timed::Number(k))
        | (Mul, Timed::Number(k), Timed::Duration(a)) => Timed::Duration(a.scaled(k)),
        (Div, Timed::Duration(a), Timed::Number(k)) => {
            if k == 0.0 {
                return Err(zero_divisor());
            }
            Timed::Duration(a.scaled(1.0 / k))
        }
        (Div, Timed::Stretch, Timed::Number(k)) => {
            if k == 0.0 {
                return Err(zero_divisor());
            }
            Timed::Stretch
        }
        (Div, Timed::Duration(a), Timed::Duration(b)) => {
            let (num, den) = if a.dt == 0.0 && b.dt == 0.0 {
                (a.ns, b.ns)
            } else if a.ns == 0.0 && b.ns == 0.0 {
                (a.dt, b.dt)
            } else {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "a duration ratio mixing `dt` with SI units, which only a \
                                backend's sample period relates"
                        .into(),
                    line,
                });
            };
            if den == 0.0 {
                return Err(zero_divisor());
            }
            Timed::Number(finite(line, op.spelling(), num / den)?)
        }
        (Div, Timed::Stretch, Timed::Duration(_) | Timed::Stretch)
        | (Div, Timed::Duration(_), Timed::Stretch) => {
            return Err(PrismError::UnsupportedConstruct {
                construct: "a ratio over a stretch, whose length only a scheduler fixes".into(),
                line,
            });
        }
        _ => {
            return Err(PrismError::Parse {
                line,
                message: format!(
                    "`{}` between {} and {}",
                    op.spelling(),
                    left.describe(),
                    right.describe()
                ),
            });
        }
    })
}

fn divide(operation: &str, left: f64, right: f64, line: usize) -> Result<f64> {
    if right == 0.0 {
        return Err(PrismError::Parse {
            line,
            message: format!("{operation} by zero in angle expression"),
        });
    }
    Ok(if operation == "division" {
        left / right
    } else {
        left % right
    })
}

fn resolve(name: &str, line: usize, vars: Option<&HashMap<&str, f64>>) -> Result<f64> {
    match name {
        "pi" | "\u{3c0}" => return Ok(std::f64::consts::PI),
        "tau" | "\u{3c4}" => return Ok(std::f64::consts::TAU),
        "euler" | "e" => return Ok(std::f64::consts::E),
        "true" => return Ok(1.0),
        "false" => return Ok(0.0),
        _ => {}
    }
    if let Some(value) = vars.and_then(|vars| vars.get(name)) {
        return Ok(*value);
    }
    Err(PrismError::Parse {
        line,
        message: format!("unknown identifier `{name}` in expression"),
    })
}

/// Apply an OpenQASM builtin. The spec spellings (`arcsin`, `ceiling`, `log`)
/// are the primary names; the shorter C ones are accepted too, since exported
/// and hand-written sources use both.
fn apply(name: &str, args: &[f64], line: usize) -> Result<f64> {
    let arity = |wanted: usize| -> Result<()> {
        if args.len() == wanted {
            Ok(())
        } else {
            Err(PrismError::Parse {
                line,
                message: format!("`{name}` takes {wanted} argument(s), got {}", args.len()),
            })
        }
    };
    let value = match name {
        "mod" => {
            arity(2)?;
            args[0] % args[1]
        }
        "pow" => {
            arity(2)?;
            args[0].powf(args[1])
        }
        _ => {
            arity(1)?;
            let arg = args[0];
            match name {
                "sin" => arg.sin(),
                "cos" => arg.cos(),
                "tan" => arg.tan(),
                "arcsin" | "asin" => arg.asin(),
                "arccos" | "acos" => arg.acos(),
                "arctan" | "atan" => arg.atan(),
                "sqrt" => arg.sqrt(),
                "exp" => arg.exp(),
                "log" | "ln" => arg.ln(),
                "log2" => arg.log2(),
                "abs" => arg.abs(),
                "ceiling" | "ceil" => arg.ceil(),
                "floor" => arg.floor(),
                "popcount" => popcount(arg, line)?,
                _ => {
                    return Err(PrismError::Parse {
                        line,
                        message: format!("unknown function `{name}` in expression"),
                    });
                }
            }
        }
    };
    finite(line, name, value)
}

/// Set bits of a non-negative integer. The argument arrives as the `f64` every
/// expression evaluates to, so a fractional or negative one is rejected rather
/// than truncated.
fn popcount(arg: f64, line: usize) -> Result<f64> {
    if arg < 0.0 || arg.fract() != 0.0 || arg > u64::MAX as f64 {
        return Err(PrismError::Parse {
            line,
            message: format!("`popcount` takes a non-negative integer, got {arg}"),
        });
    }
    Ok(f64::from((arg as u64).count_ones()))
}

/// Reject a result no angle can carry, naming what produced it.
fn finite(line: usize, source: &str, value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PrismError::Parse {
            line,
            message: format!("`{source}` produced the non-finite value {value}"),
        })
    }
}

#[cfg(test)]
#[path = "expr_tests.rs"]
mod expr_tests;
