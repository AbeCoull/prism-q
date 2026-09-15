//! Expression syntax tree, its parser, and its evaluator.
//!
//! Parsing and evaluation are separate so a body that runs many times, a `for`
//! body above all, is parsed once and evaluated per pass.

use std::collections::HashMap;
use std::fmt;

use super::lexer::Kind;
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

#[derive(Clone, Debug)]
pub(crate) enum Expr<'a> {
    Number(f64),
    Ident(&'a str),
    Negate(Box<Expr<'a>>),
    Binary {
        op: BinaryOp,
        left: Box<Expr<'a>>,
        right: Box<Expr<'a>>,
    },
    /// Boxed because a call is the one wide variant and a leaf is the common
    /// one, so the width of every node would otherwise follow the call.
    Call(Box<Call<'a>>),
}

#[derive(Clone, Debug)]
pub(crate) struct Call<'a> {
    pub name: &'a str,
    pub args: Vec<Expr<'a>>,
}

impl<'a> Expr<'a> {
    /// Names this expression reads, which is what decides whether it depends
    /// on an `input` slot.
    pub(crate) fn mentions(&self, name: &str) -> bool {
        match self {
            Expr::Number(_) => false,
            Expr::Ident(ident) => *ident == name,
            Expr::Negate(inner) => inner.mentions(name),
            Expr::Binary { left, right, .. } => left.mentions(name) || right.mentions(name),
            Expr::Call(call) => call.args.iter().any(|arg| arg.mentions(name)),
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
            Ok(Expr::Number(number(token.text, token.line)?))
        }
        Kind::Ident => {
            let name = stream.advance().text;
            if !stream.eat(Kind::LParen) {
                return Ok(Expr::Ident(name));
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
    let cleaned: String = text.chars().filter(|c| *c != '_').collect();
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

/// Fold an expression to its value, resolving names against `vars`.
/// Evaluate a tree to the single number an angle, an index or a bound takes.
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
        Expr::Ident(name) => resolve(name, line, vars),
        Expr::Negate(inner) => Ok(-evaluate(inner, line, vars)?),
        Expr::Binary { op, left, right } => {
            let left = evaluate(left, line, vars)?;
            let right = evaluate(right, line, vars)?;
            match op {
                BinaryOp::Add => Ok(left + right),
                BinaryOp::Sub => Ok(left - right),
                BinaryOp::Mul => Ok(left * right),
                BinaryOp::Div => divide("division", left, right, line),
                BinaryOp::Rem => divide("modulo", left, right, line),
                BinaryOp::Pow => finite(line, op.spelling(), left.powf(right)),
            }
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
