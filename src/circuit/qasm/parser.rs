//! Tokens to a syntax tree.
//!
//! The grammar is decided here and nowhere else: no later pass rescans text to
//! find where a statement, a block or an argument ended.

use super::ast::{
    Argument, AssignOp, Block, CmpOp, Condition, Conditional, DefParam, ForRange, Index, Measure,
    Modifier, Operand, OperandName, Range, RegisterKind, Stmt, StmtKind, SwitchArm,
};
use super::expr::{self, Expr};
use super::lexer::{Kind, Token};
use super::stream::Stream;
use crate::error::{PrismError, Result};

/// Keywords that open a classical declaration. `complex`, `duration` and
/// `stretch` parse and are declined by the evaluator, which is where the
/// reason belongs.
const DECLARATION_TYPES: &[&str] = &[
    "int", "uint", "float", "angle", "bool", "complex", "duration", "stretch",
];

/// Keywords the language has and this parser does not implement.
const UNSUPPORTED: &[&str] = &[
    "defcal", "extern", "opaque", "while", "return", "break", "continue", "else",
];

pub(crate) fn parse_program<'a>(tokens: &[Token<'a>]) -> Result<Block<'a>> {
    let mut stream = Stream::new(tokens);
    let mut out = Block::new();
    while !stream.at_end() {
        out.push(statement(&mut stream)?);
    }
    Ok(out)
}

fn statement<'a>(stream: &mut Stream<'_, 'a>) -> Result<Stmt<'a>> {
    let line = stream.line();
    let kind = statement_kind(stream)?;
    Ok(Stmt { line, kind })
}

fn statement_kind<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    if stream.kind() == Kind::Pragma {
        return Ok(StmtKind::Pragma(stream.advance().text));
    }
    if stream.eat(Kind::Semicolon) {
        return Ok(StmtKind::Empty);
    }
    if stream.kind() != Kind::Ident {
        return Err(stream.expected("a statement"));
    }

    let word = stream.peek().text;
    if let Some(keyword) = UNSUPPORTED.iter().find(|entry| **entry == word) {
        return Err(PrismError::UnsupportedConstruct {
            construct: (*keyword).to_string(),
            line: stream.line(),
        });
    }

    match word {
        "OPENQASM" => version(stream),
        "include" => include(stream),
        "qubit" => oq3_register(stream, RegisterKind::Qubit),
        "bit" => oq3_register(stream, RegisterKind::Classical),
        "qreg" => legacy_register(stream, RegisterKind::Qubit),
        "creg" => legacy_register(stream, RegisterKind::Classical),
        "input" | "output" => port(stream),
        "let" => alias(stream),
        "measure" => measure_arrow(stream),
        "reset" => reset(stream),
        "barrier" => barrier(stream),
        "if" => conditional(stream),
        "for" => for_loop(stream),
        "switch" => switch(stream),
        "gate" => gate_def(stream),
        "def" => def_def(stream),
        "box" => {
            stream.advance();
            Ok(StmtKind::Box(braced_block(stream)?))
        }
        "const" => declaration(stream),
        other if DECLARATION_TYPES.contains(&other) => declaration(stream),
        _ => call_or_assignment(stream),
    }
}

fn version<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let token = stream.peek();
    if !matches!(token.kind, Kind::Int | Kind::Float) {
        return Err(stream.expected("an OpenQASM version"));
    }
    stream.advance();
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Version(token.text))
}

fn include<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    stream.expect(Kind::Str)?;
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Include)
}

/// `qubit[4] q;` and `bit c;`, where the size precedes the name.
fn oq3_register<'a>(stream: &mut Stream<'_, 'a>, kind: RegisterKind) -> Result<StmtKind<'a>> {
    stream.advance();
    let size = if stream.eat(Kind::LBracket) {
        let size = expr::parse(stream)?;
        stream.expect(Kind::RBracket)?;
        Some(size)
    } else {
        None
    };
    let name = stream.expect_ident()?;
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::RegisterDecl { kind, name, size })
}

/// `qreg q[4];`, where the size follows the name.
fn legacy_register<'a>(stream: &mut Stream<'_, 'a>, kind: RegisterKind) -> Result<StmtKind<'a>> {
    stream.advance();
    let name = stream.expect_ident()?;
    stream.expect(Kind::LBracket)?;
    let size = expr::parse(stream)?;
    stream.expect(Kind::RBracket)?;
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::RegisterDecl {
        kind,
        name,
        size: Some(size),
    })
}

/// `input float[64] theta;` and `output bit[4] c;`.
fn port<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    let direction = stream.advance().text;
    let (ty, size) = typename(stream)?;
    let name = stream.expect_ident()?;
    stream.expect(Kind::Semicolon)?;
    if direction == "input" {
        return Ok(StmtKind::InputDecl { ty, name });
    }
    Ok(StmtKind::OutputDecl { ty, name, size })
}

/// A type name with its optional width, which every declaration shares.
fn typename<'a>(stream: &mut Stream<'_, 'a>) -> Result<(&'a str, Option<Expr<'a>>)> {
    let ty = stream.expect_ident()?;
    if !stream.eat(Kind::LBracket) {
        return Ok((ty, None));
    }
    let width = expr::parse(stream)?;
    stream.expect(Kind::RBracket)?;
    Ok((ty, Some(width)))
}

fn declaration<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    let constant = stream.eat_keyword("const");
    let (ty, _width) = typename(stream)?;
    let name = stream.expect_ident()?;
    let value = if stream.eat(Kind::Assign) {
        Some(expr::parse(stream)?)
    } else {
        None
    };
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::ClassicalDecl {
        constant,
        ty,
        name,
        value,
    })
}

/// `let a = q[0:1] ++ q[3];`
fn alias<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let name = stream.expect_ident()?;
    stream.expect(Kind::Assign)?;
    let mut sources = vec![operand(stream)?];
    while stream.eat(Kind::Concat) {
        sources.push(operand(stream)?);
    }
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Alias { name, sources })
}

/// `measure q[0] -> c[0];`, the OpenQASM 2 spelling.
fn measure_arrow<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let source = operand(stream)?;
    if stream.kind() != Kind::Arrow {
        return Err(stream.expected("`->` in `measure qubit -> bit`"));
    }
    stream.advance();
    let target = operand(stream)?;
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Measure(Box::new(Measure { source, target })))
}

fn reset<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let targets = operand_list(stream, "reset")?;
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Reset { targets })
}

fn barrier<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let targets = operand_list(stream, "barrier")?;
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Barrier { targets })
}

fn operand_list<'a>(stream: &mut Stream<'_, 'a>, what: &str) -> Result<Vec<Operand<'a>>> {
    if stream.kind() == Kind::Semicolon {
        return Err(stream.expected(&format!("a qubit after `{what}`")));
    }
    let mut out = vec![operand(stream)?];
    while stream.eat(Kind::Comma) {
        out.push(operand(stream)?);
    }
    Ok(out)
}

fn operand<'a>(stream: &mut Stream<'_, 'a>) -> Result<Operand<'a>> {
    let token = stream.peek();
    let name = match token.kind {
        Kind::Physical => {
            stream.advance();
            let index = token.text.parse::<usize>().map_err(|_| PrismError::Parse {
                line: token.line,
                message: format!("`${}` is not a qubit index", token.text),
            })?;
            OperandName::Physical(index)
        }
        Kind::Ident => OperandName::Register(stream.advance().text),
        _ => return Err(stream.expected("a qubit or bit reference")),
    };
    let index = if stream.eat(Kind::LBracket) {
        let index = subscript(stream)?;
        stream.expect(Kind::RBracket)?;
        Some(index)
    } else {
        None
    };
    Ok(Operand {
        name,
        index,
        line: token.line,
    })
}

/// What sits between `[` and `]`: one index, an inclusive range with an
/// optional step in the middle, or an explicit set.
fn subscript<'a>(stream: &mut Stream<'_, 'a>) -> Result<Index<'a>> {
    if stream.eat(Kind::LBrace) {
        let mut entries = vec![expr::parse(stream)?];
        while stream.eat(Kind::Comma) {
            entries.push(expr::parse(stream)?);
        }
        stream.expect(Kind::RBrace)?;
        return Ok(Index::Set(entries));
    }

    let first = optional_bound(stream)?;
    if !stream.eat(Kind::Colon) {
        return match first {
            Some(index) => Ok(Index::Single(index)),
            None => Err(stream.expected("an index")),
        };
    }
    let second = optional_bound(stream)?;
    if !stream.eat(Kind::Colon) {
        return Ok(Index::Range(Box::new(Range {
            start: first,
            step: None,
            stop: second,
        })));
    }
    Ok(Index::Range(Box::new(Range {
        start: first,
        step: second,
        stop: optional_bound(stream)?,
    })))
}

/// One end of a range, absent when the source left it open.
fn optional_bound<'a>(stream: &mut Stream<'_, 'a>) -> Result<Option<Expr<'a>>> {
    if matches!(stream.kind(), Kind::Colon | Kind::RBracket) {
        return Ok(None);
    }
    Ok(Some(expr::parse(stream)?))
}

fn conditional<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    stream.expect(Kind::LParen)?;
    let condition = parse_condition(stream)?;
    stream.expect(Kind::RParen)?;
    let then_body = body(stream)?;
    let else_body = if stream.eat_keyword("else") {
        Some(body(stream)?)
    } else {
        None
    };
    Ok(StmtKind::If(Box::new(Conditional {
        condition,
        then_body,
        else_body,
    })))
}

/// A braced block, or the single statement that stands in for one.
fn body<'a>(stream: &mut Stream<'_, 'a>) -> Result<Block<'a>> {
    if stream.kind() == Kind::LBrace {
        return braced_block(stream);
    }
    Ok(vec![statement(stream)?])
}

fn braced_block<'a>(stream: &mut Stream<'_, 'a>) -> Result<Block<'a>> {
    stream.expect(Kind::LBrace)?;
    let mut out = Block::new();
    while !stream.eat(Kind::RBrace) {
        if stream.at_end() {
            return Err(stream.expected("`}` closing a block"));
        }
        out.push(statement(stream)?);
    }
    Ok(out)
}

fn parse_condition<'a>(stream: &mut Stream<'_, 'a>) -> Result<Condition<'a>> {
    if stream.eat(Kind::LParen) {
        let inner = parse_condition(stream)?;
        stream.expect(Kind::RParen)?;
        let Some((op, rhs)) = comparison(stream)? else {
            return Ok(inner);
        };
        // Only a parity reads as a value a comparison can stand against.
        let Condition::Parity {
            bits,
            compare: None,
        } = inner
        else {
            return Err(PrismError::Parse {
                line: stream.line(),
                message: "only a parity condition compares against a literal".into(),
            });
        };
        return Ok(Condition::Parity {
            bits,
            compare: Some((op, rhs)),
        });
    }
    if stream.eat(Kind::Bang) {
        return Ok(Condition::Negated(operand(stream)?));
    }
    let first = operand(stream)?;
    if stream.kind() == Kind::Caret {
        let mut bits = vec![first];
        while stream.eat(Kind::Caret) {
            bits.push(operand(stream)?);
        }
        let compare = comparison(stream)?;
        return Ok(Condition::Parity { bits, compare });
    }
    match comparison(stream)? {
        Some((op, rhs)) => Ok(Condition::Compare {
            lhs: first,
            op,
            rhs,
        }),
        None => Ok(Condition::Truthy(first)),
    }
}

fn comparison<'a>(stream: &mut Stream<'_, 'a>) -> Result<Option<(CmpOp, Expr<'a>)>> {
    let op = match stream.kind() {
        Kind::EqEq => CmpOp::Equal,
        Kind::NotEq => CmpOp::NotEqual,
        _ => return Ok(None),
    };
    stream.advance();
    Ok(Some((op, expr::parse(stream)?)))
}

fn for_loop<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    // `for int i in ...` and `for i in ...` both stand; the type is decoration
    // the loop bounds already carry.
    if stream.kind() == Kind::Ident && matches!(stream.peek().text, "int" | "uint") {
        let mark = stream.mark();
        typename(stream)?;
        if stream.kind() != Kind::Ident {
            stream.rewind(mark);
        }
    }
    let variable = stream.expect_ident()?;
    if !stream.eat_keyword("in") {
        return Err(stream.expected("`in` after a loop variable"));
    }
    let range = for_range(stream)?;
    let body = braced_block(stream)?;
    Ok(StmtKind::For {
        variable,
        range,
        body,
    })
}

fn for_range<'a>(stream: &mut Stream<'_, 'a>) -> Result<ForRange<'a>> {
    if stream.eat(Kind::LBrace) {
        let mut entries = vec![expr::parse(stream)?];
        while stream.eat(Kind::Comma) {
            entries.push(expr::parse(stream)?);
        }
        stream.expect(Kind::RBrace)?;
        return Ok(ForRange::Set(entries));
    }
    if !stream.eat(Kind::LBracket) {
        return Err(PrismError::UnsupportedConstruct {
            construct: format!(
                "for loop range starting with {} (only `[start:stop]`, `[start:step:stop]`, \
                 or `{{a,b,c}}` supported)",
                stream.peek().describe()
            ),
            line: stream.line(),
        });
    }
    let start = expr::parse(stream)?;
    stream.expect(Kind::Colon)?;
    let second = expr::parse(stream)?;
    let (step, stop) = if stream.eat(Kind::Colon) {
        (Some(second), expr::parse(stream)?)
    } else {
        (None, second)
    };
    stream.expect(Kind::RBracket)?;
    Ok(ForRange::Range { start, step, stop })
}

fn switch<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    stream.expect(Kind::LParen)?;
    let operand = operand(stream)?;
    stream.expect(Kind::RParen)?;
    stream.expect(Kind::LBrace)?;
    let mut arms = Vec::new();
    while !stream.eat(Kind::RBrace) {
        let line = stream.line();
        if stream.eat_keyword("case") {
            let mut labels = vec![expr::parse(stream)?];
            while stream.eat(Kind::Comma) {
                labels.push(expr::parse(stream)?);
            }
            arms.push(SwitchArm {
                labels: Some(labels),
                body: braced_block(stream)?,
                line,
            });
        } else if stream.eat_keyword("default") {
            arms.push(SwitchArm {
                labels: None,
                body: braced_block(stream)?,
                line,
            });
        } else {
            return Err(PrismError::Parse {
                line,
                message: format!(
                    "expected `case` or `default` in `switch` body, got {}",
                    stream.peek().describe()
                ),
            });
        }
    }
    Ok(StmtKind::Switch { operand, arms })
}

fn gate_def<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let name = stream.expect_ident()?;
    let mut params = Vec::new();
    if stream.eat(Kind::LParen) {
        while !stream.eat(Kind::RParen) {
            params.push(stream.expect_ident()?);
            if !stream.eat(Kind::Comma) {
                stream.expect(Kind::RParen)?;
                break;
            }
        }
    }
    let mut qubits = vec![stream.expect_ident()?];
    while stream.eat(Kind::Comma) {
        qubits.push(stream.expect_ident()?);
    }
    let body = braced_block(stream)?;
    Ok(StmtKind::GateDef {
        name,
        params,
        qubits,
        body,
    })
}

fn def_def<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    let line = stream.line();
    stream.advance();
    let name = stream.expect_ident()?;
    stream.expect(Kind::LParen)?;
    let mut args = Vec::new();
    while !stream.eat(Kind::RParen) {
        let (ty, _width) = typename(stream)?;
        let arg_name = stream.expect_ident()?;
        args.push(match ty {
            "qubit" => DefParam::Qubit(arg_name),
            "int" | "uint" => DefParam::Value {
                name: arg_name,
                integral: true,
            },
            "float" | "angle" | "complex" | "duration" | "stretch" => DefParam::Value {
                name: arg_name,
                integral: false,
            },
            "bit" | "creg" => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "classical bit parameters in def `{name}` (V1 supports unitary \
                         subroutines only)"
                    ),
                    line,
                });
            }
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("def parameter type `{other}`"),
                    line,
                });
            }
        });
        if !stream.eat(Kind::Comma) {
            stream.expect(Kind::RParen)?;
            break;
        }
    }
    if stream.kind() == Kind::Arrow {
        return Err(PrismError::UnsupportedConstruct {
            construct: "def with return type".to_string(),
            line,
        });
    }
    let body = braced_block(stream)?;
    Ok(StmtKind::DefDef { name, args, body })
}

/// A gate application, a `def` call, or an assignment, which all open with a
/// name. Only what follows the name tells them apart.
fn call_or_assignment<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    if let Some(kind) = try_assignment(stream)? {
        return Ok(kind);
    }
    let modifiers = modifier_chain(stream)?;
    let name = stream.expect_ident()?;
    let mut params = Vec::new();
    if stream.eat(Kind::LParen) {
        while !stream.eat(Kind::RParen) {
            params.push(argument(stream)?);
            if !stream.eat(Kind::Comma) {
                stream.expect(Kind::RParen)?;
                break;
            }
        }
    }
    let mut operands = Vec::new();
    if stream.kind() != Kind::Semicolon {
        operands = operand_list(stream, name)?;
    }
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Call {
        modifiers,
        name,
        params,
        operands,
    })
}

/// `c = measure q;`, `n = n + 1;` and `n += 1;`, told from a gate call by the
/// assignment that follows the target.
fn try_assignment<'a>(stream: &mut Stream<'_, 'a>) -> Result<Option<StmtKind<'a>>> {
    let mark = stream.mark();
    let Ok(target) = operand(stream) else {
        stream.rewind(mark);
        return Ok(None);
    };
    let op = match stream.kind() {
        Kind::Assign => None,
        Kind::AddAssign => Some(AssignOp::Add),
        Kind::SubAssign => Some(AssignOp::Sub),
        Kind::MulAssign => Some(AssignOp::Mul),
        Kind::DivAssign => Some(AssignOp::Div),
        Kind::ModAssign => Some(AssignOp::Rem),
        _ => {
            stream.rewind(mark);
            return Ok(None);
        }
    };
    stream.advance();

    if op.is_none() && stream.eat_keyword("measure") {
        let source = operand(stream)?;
        stream.expect(Kind::Semicolon)?;
        return Ok(Some(StmtKind::Measure(Box::new(Measure {
            source,
            target,
        }))));
    }

    let value = expr::parse(stream)?;
    stream.expect(Kind::Semicolon)?;
    let Some(name) = target.register().filter(|_| target.index.is_none()) else {
        return Err(PrismError::Parse {
            line: target.line,
            message: format!(
                "`{}` is not a name an assignment can write",
                target.describe()
            ),
        });
    };
    Ok(Some(StmtKind::Assign {
        target: name,
        op,
        value,
    }))
}

/// `inv @`, `pow(k) @`, `ctrl @` and `negctrl @`, in any order and chainable.
fn modifier_chain<'a>(stream: &mut Stream<'_, 'a>) -> Result<Vec<Modifier<'a>>> {
    let mut out = Vec::new();
    loop {
        if stream.kind() != Kind::Ident {
            return Ok(out);
        }
        let word = stream.peek().text;
        if !matches!(word, "inv" | "ctrl" | "negctrl" | "pow") {
            // A name nobody modifies with is still a modifier when `@` follows
            // it, and a gate otherwise.
            if stream.peek_at(1).kind == Kind::At {
                return Err(PrismError::UnsupportedConstruct {
                    construct: word.to_string(),
                    line: stream.line(),
                });
            }
            return Ok(out);
        }
        let mark = stream.mark();
        stream.advance();
        let mut exponent = None;
        if stream.kind() == Kind::LParen {
            stream.advance();
            let parsed = expr::parse(stream);
            match parsed {
                Ok(value) if stream.eat(Kind::RParen) => exponent = Some(value),
                _ => {
                    stream.rewind(mark);
                    return Ok(out);
                }
            }
        }
        if !stream.eat(Kind::At) {
            stream.rewind(mark);
            return Ok(out);
        }
        out.push(match (word, exponent) {
            ("inv", None) => Modifier::Inv,
            ("ctrl", None) => Modifier::Ctrl { negated: false },
            ("negctrl", None) => Modifier::Ctrl { negated: true },
            ("pow", Some(exponent)) => Modifier::Pow(exponent),
            _ => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: word.to_string(),
                    line: stream.line(),
                });
            }
        });
    }
}

/// One entry inside a call's parentheses. A subscript settles it as a qubit;
/// anything else parses as a value, and a bare name is read as whichever the
/// declaration asks for.
fn argument<'a>(stream: &mut Stream<'_, 'a>) -> Result<Argument<'a>> {
    if stream.kind() == Kind::Physical
        || (stream.kind() == Kind::Ident && stream.peek_at(1).kind == Kind::LBracket)
    {
        return Ok(Argument::Operand(operand(stream)?));
    }
    Ok(Argument::Value(expr::parse(stream)?))
}

/// Parse a comma-separated operand list standing on its own, which is the
/// shape a `#pragma` body writes its targets in.
pub(crate) fn parse_operands<'a>(tokens: &[Token<'a>]) -> Result<Vec<Operand<'a>>> {
    let mut stream = Stream::new(tokens);
    let operands = operand_list(&mut stream, "a qubit reference")?;
    if !stream.at_end() {
        return Err(stream.expected("the end of the qubit list"));
    }
    Ok(operands)
}

#[cfg(test)]
#[path = "parser_tests.rs"]
mod parser_tests;
