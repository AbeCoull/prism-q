//! Tokens to a syntax tree.
//!
//! The grammar is decided here and nowhere else: no later pass rescans text to
//! find where a statement, a block or an argument ended.

use super::ast::{
    Argument, ArrayDecl, ArrayInit, ArrayShape, AssignOp, BitInit, Block, CallAssign, CmpOp,
    Condition, Conditional, DefParam, DefType, ElementAssign, ForRange, Index, Measure, Modifier,
    Operand, OperandName, Range, RegisterKind, ReturnValue, Stmt, StmtKind, SwitchArm,
};
use super::expr::{self, Expr};
use super::lexer::{Kind, Token};
use super::stream::Stream;
use crate::error::{PrismError, Result};
use smallvec::SmallVec;

/// Keywords that open a classical declaration.
const DECLARATION_TYPES: &[&str] = &[
    "int", "uint", "float", "angle", "bool", "complex", "duration", "stretch",
];

/// Keywords the language has and this parser declines, with the text each decline
/// carries. Declining by name returns `UnsupportedConstruct` where the operands
/// would otherwise give a bare syntax error (`while (c[0]) { }` reads as a
/// malformed gate call). A dynamic parse reads `while` instead.
const UNSUPPORTED: &[(&str, &str)] = &[
    ("extern", "extern"),
    ("opaque", "opaque"),
    ("while", "while"),
    ("else", "else"),
];

/// Deepest nesting an array initializer may have, which bounds the recursion
/// that reads it.
const MAX_INITIALIZER_DEPTH: usize = 32;

pub(crate) fn parse_program<'a>(tokens: &[Token<'a>]) -> Result<Block<'a>> {
    program(Stream::new(tokens), tokens.len())
}

/// [`parse_program`] with `while`, and conditions and
/// values over the full classical operator set.
pub(crate) fn parse_dynamic_program<'a>(tokens: &[Token<'a>]) -> Result<Block<'a>> {
    program(Stream::dynamic(tokens), tokens.len())
}

fn program<'a>(mut stream: Stream<'_, 'a>, num_tokens: usize) -> Result<Block<'a>> {
    // A gate statement runs six to twelve tokens.
    let mut out = Block::with_capacity(num_tokens / 6);
    while !stream.at_end() {
        statement_into(&mut stream, &mut out)?;
    }
    Ok(out)
}

/// Append the next statement to `out`. A `bit` declaration initialized by a
/// measurement appends two, the declaration and the measurement.
fn statement_into<'a>(stream: &mut Stream<'_, 'a>, out: &mut Block<'a>) -> Result<()> {
    if stream.is_keyword("bit") {
        return bit_declaration(stream, out);
    }
    out.push(statement(stream)?);
    Ok(())
}

fn statement<'a>(stream: &mut Stream<'_, 'a>) -> Result<Stmt<'a>> {
    stream.nested(|stream| {
        let line = stream.line();
        let kind = statement_kind(stream)?;
        Ok(Stmt { line, kind })
    })
}

fn statement_kind<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    if stream.kind() == Kind::Pragma {
        return Ok(StmtKind::Pragma(stream.advance().text));
    }
    if stream.eat(Kind::Semicolon) {
        return Ok(StmtKind::Empty);
    }
    if opens_expression(stream) {
        let value = expr::parse(stream)?;
        stream.expect(Kind::Semicolon)?;
        return Ok(StmtKind::Expr(value));
    }
    if stream.kind() != Kind::Ident {
        return Err(stream.expected("a statement"));
    }

    let word = stream.peek().text;
    let reads_loops = stream.is_dynamic() && word == "while";
    if let Some((_, construct)) = UNSUPPORTED
        .iter()
        .find(|(keyword, _)| *keyword == word && !reads_loops)
    {
        return Err(PrismError::UnsupportedConstruct {
            construct: (*construct).to_string(),
            line: stream.line(),
        });
    }

    match word {
        "OPENQASM" => version(stream),
        "include" => include(stream),
        "qubit" => qubit_register(stream),
        "qreg" => legacy_register(stream, RegisterKind::Qubit),
        "creg" => legacy_register(stream, RegisterKind::Classical),
        "input" | "output" => port(stream),
        "let" => alias(stream),
        "measure" => measure_arrow(stream),
        "reset" => reset(stream),
        "barrier" => barrier(stream),
        "delay" => delay(stream),
        "nop" => nop(stream),
        "defcalgrammar" => {
            stream.advance();
            stream.expect(Kind::Str)?;
            stream.expect(Kind::Semicolon)?;
            Ok(StmtKind::Empty)
        }
        "cal" | "defcal" => calibration(stream),
        "if" => conditional(stream),
        "for" => for_loop(stream),
        "while" => while_loop(stream),
        "break" | "continue" => {
            stream.advance();
            stream.expect(Kind::Semicolon)?;
            Ok(if word == "break" {
                StmtKind::Break
            } else {
                StmtKind::Continue
            })
        }
        "end" if stream.peek_at(1).kind == Kind::Semicolon => {
            stream.advance();
            stream.advance();
            Ok(StmtKind::End)
        }
        "switch" => switch(stream),
        "gate" => gate_def(stream),
        "def" => def_def(stream),
        "return" => return_statement(stream),
        "box" => boxed(stream),
        "array" => array_declaration(stream, false),
        "const" => declaration(stream),
        other if DECLARATION_TYPES.contains(&other) => declaration(stream),
        _ => call_or_assignment(stream),
    }
}

/// Whether the statement at the cursor is an expression statement: one that
/// opens with a literal, a parenthesis or a unary operator, or a name whose call
/// or index groups an operator follows. A gate call never puts an operator
/// there, so `rx(0.5) q;` stays a call without being read twice.
fn opens_expression(stream: &Stream<'_, '_>) -> bool {
    match stream.kind() {
        Kind::Int
        | Kind::Float
        | Kind::Duration
        | Kind::LParen
        | Kind::Minus
        | Kind::Plus
        | Kind::Bang
        | Kind::Tilde => return true,
        Kind::Ident => {}
        _ => return false,
    }
    if matches!(
        stream.peek().text,
        "return" | "if" | "while" | "for" | "switch" | "box"
    ) {
        return false;
    }
    let mut ahead = 1;
    while matches!(stream.peek_at(ahead).kind, Kind::LParen | Kind::LBracket) {
        let mut depth = 0usize;
        loop {
            match stream.peek_at(ahead).kind {
                Kind::LParen | Kind::LBracket => depth += 1,
                Kind::RParen | Kind::RBracket => depth -= 1,
                Kind::Eof => return false,
                _ => {}
            }
            ahead += 1;
            if depth == 0 {
                break;
            }
        }
    }
    matches!(
        stream.peek_at(ahead).kind,
        Kind::EqEq
            | Kind::NotEq
            | Kind::Lt
            | Kind::Le
            | Kind::Gt
            | Kind::Ge
            | Kind::Plus
            | Kind::Minus
            | Kind::Star
            | Kind::Slash
            | Kind::Percent
            | Kind::Pow
            | Kind::Amp
            | Kind::Pipe
            | Kind::Caret
            | Kind::AndAnd
            | Kind::OrOr
            | Kind::Shl
            | Kind::Shr
    )
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

/// `qubit[4] q;`, where the size precedes the name.
fn qubit_register<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let size = register_size(stream)?;
    let name = stream.expect_ident()?;
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::RegisterDecl {
        kind: RegisterKind::Qubit,
        name,
        size,
        init: None,
    })
}

/// `bit[2] c;`, with an optional initializer: `= measure q;`, a bit string, or a
/// value.
fn bit_declaration<'a>(stream: &mut Stream<'_, 'a>, out: &mut Block<'a>) -> Result<()> {
    let line = stream.line();
    stream.advance();
    let size = register_size(stream)?;
    let name = stream.expect_ident()?;
    let mut source = None;
    let mut init = None;
    if stream.eat(Kind::Assign) {
        if stream.eat_keyword("measure") {
            source = Some(operand(stream)?);
        } else if stream.kind() == Kind::Str {
            init = Some(Box::new(BitInit::Bits(stream.advance().text)));
        } else {
            init = Some(Box::new(BitInit::Value(expr::parse(stream)?)));
        }
    }
    stream.expect(Kind::Semicolon)?;
    out.push(Stmt {
        line,
        kind: StmtKind::RegisterDecl {
            kind: RegisterKind::Classical,
            name,
            size,
            init,
        },
    });
    if let Some(source) = source {
        let target = Operand {
            name: OperandName::Register(name),
            index: None,
            line,
        };
        out.push(Stmt {
            line,
            kind: StmtKind::Measure(Box::new(Measure { source, target })),
        });
    }
    Ok(())
}

fn register_size<'a>(stream: &mut Stream<'_, 'a>) -> Result<Option<Expr<'a>>> {
    if !stream.eat(Kind::LBracket) {
        return Ok(None);
    }
    let size = expr::parse(stream)?;
    stream.expect(Kind::RBracket)?;
    Ok(Some(size))
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
        init: None,
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

/// A type name with its optional width, which every declaration shares. The
/// width of `complex[float[n]]` is the `n` of its components.
fn typename<'a>(stream: &mut Stream<'_, 'a>) -> Result<(&'a str, Option<Expr<'a>>)> {
    let ty = stream.expect_ident()?;
    if !stream.eat(Kind::LBracket) {
        return Ok((ty, None));
    }
    if ty == "complex" {
        let width = expr::complex_component(stream)?;
        stream.expect(Kind::RBracket)?;
        return Ok((ty, width));
    }
    let width = expr::parse(stream)?;
    stream.expect(Kind::RBracket)?;
    Ok((ty, Some(width)))
}

fn declaration<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    let constant = stream.eat_keyword("const");
    if stream.is_keyword("array") {
        return array_declaration(stream, constant);
    }
    let (ty, width) = typename(stream)?;
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
        width,
        name,
        value,
    })
}

/// `array[int[32], 2, 3] a = {{1, 2, 3}, {4, 5, 6}};`, the element type first and
/// one size per dimension after it.
fn array_declaration<'a>(stream: &mut Stream<'_, 'a>, constant: bool) -> Result<StmtKind<'a>> {
    stream.advance();
    stream.expect(Kind::LBracket)?;
    let (ty, width) = typename(stream)?;
    let mut dims = Vec::new();
    while stream.eat(Kind::Comma) {
        dims.push(expr::parse(stream)?);
    }
    if dims.is_empty() {
        return Err(stream.expected("`,` and a size after the element type"));
    }
    stream.expect(Kind::RBracket)?;
    let name = stream.expect_ident()?;
    let init = if stream.eat(Kind::Assign) {
        Some(array_init(stream, 0)?)
    } else {
        None
    };
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::ArrayDecl(Box::new(ArrayDecl {
        constant,
        ty,
        width,
        dims,
        name,
        init,
    })))
}

fn array_init<'a>(stream: &mut Stream<'_, 'a>, depth: usize) -> Result<ArrayInit<'a>> {
    if !stream.eat(Kind::LBrace) {
        return Ok(ArrayInit::Value(value(stream)?));
    }
    if depth >= MAX_INITIALIZER_DEPTH {
        return Err(PrismError::Parse {
            line: stream.line(),
            message: format!("array initializer nests deeper than {MAX_INITIALIZER_DEPTH}"),
        });
    }
    let mut entries = vec![array_init(stream, depth + 1)?];
    while stream.eat(Kind::Comma) {
        entries.push(array_init(stream, depth + 1)?);
    }
    stream.expect(Kind::RBrace)?;
    Ok(ArrayInit::List(entries))
}

/// The value an assignment or an array initializer writes: an expression, or
/// arrays joined with `++`.
fn value<'a>(stream: &mut Stream<'_, 'a>) -> Result<Expr<'a>> {
    let first = expr::parse(stream)?;
    if stream.kind() != Kind::Concat {
        return Ok(first);
    }
    let mut parts = vec![first];
    while stream.eat(Kind::Concat) {
        parts.push(expr::parse(stream)?);
    }
    Ok(Expr::Concat(parts))
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
    let targets = if stream.kind() == Kind::Semicolon {
        Vec::new()
    } else {
        operand_list(stream, "barrier")?
    };
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Barrier { targets })
}

/// `delay[100ns] q[0], q[1];`, where no operand means every qubit.
fn delay<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    stream.expect(Kind::LBracket)?;
    let duration = expr::parse(stream)?;
    stream.expect(Kind::RBracket)?;
    let targets = if stream.kind() == Kind::Semicolon {
        Vec::new()
    } else {
        operand_list(stream, "delay")?
    };
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Delay { duration, targets })
}

fn nop<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let targets = if stream.kind() == Kind::Semicolon {
        Vec::new()
    } else {
        operand_list(stream, "nop")?
    };
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Nop { targets })
}

/// `cal { ... }` and `defcal name(args) operands -> type { ... }`. A calibration
/// says how a gate is pulsed, not what it does, so a gate-level simulation keeps
/// the gate's own meaning and the statement reads as empty. The lexer has
/// already taken the body whole.
fn calibration<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    let keyword = stream.advance().text;
    while !stream.eat(Kind::Calibration) {
        if matches!(stream.kind(), Kind::Semicolon | Kind::Eof) {
            return Err(stream.expected(&format!("a braced body for `{keyword}`")));
        }
        stream.advance();
    }
    Ok(StmtKind::Empty)
}

/// `box { ... }`, with an optional duration designator before the body.
fn boxed<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let duration = if stream.eat(Kind::LBracket) {
        let duration = expr::parse(stream)?;
        stream.expect(Kind::RBracket)?;
        Some(duration)
    } else {
        None
    };
    Ok(StmtKind::Box {
        duration,
        body: braced_block(stream)?,
    })
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
                line: token.line as usize,
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
        line: token.line as usize,
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
    let condition = parenthesized_condition(stream)?;
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

/// `(condition)`: one of the bit and register forms when one spans the
/// parentheses, any other expression otherwise.
fn parenthesized_condition<'a>(stream: &mut Stream<'_, 'a>) -> Result<Condition<'a>> {
    stream.expect(Kind::LParen)?;
    let mark = stream.mark();
    if let Ok(condition) = parse_condition(stream)
        && stream.eat(Kind::RParen)
    {
        return Ok(condition);
    }
    stream.rewind(mark);
    let condition = Condition::Expr(expr::parse(stream)?);
    stream.expect(Kind::RParen)?;
    Ok(condition)
}

fn while_loop<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let condition = parenthesized_condition(stream)?;
    let body = body(stream)?;
    Ok(StmtKind::While { condition, body })
}

/// A braced block, or the single statement that stands in for one.
fn body<'a>(stream: &mut Stream<'_, 'a>) -> Result<Block<'a>> {
    if stream.kind() == Kind::LBrace {
        return braced_block(stream);
    }
    let mut out = Block::new();
    statement_into(stream, &mut out)?;
    Ok(out)
}

fn braced_block<'a>(stream: &mut Stream<'_, 'a>) -> Result<Block<'a>> {
    stream.expect(Kind::LBrace)?;
    let mut out = Block::new();
    while !stream.eat(Kind::RBrace) {
        if stream.at_end() {
            return Err(stream.expected("`}` closing a block"));
        }
        statement_into(stream, &mut out)?;
    }
    Ok(out)
}

fn parse_condition<'a>(stream: &mut Stream<'_, 'a>) -> Result<Condition<'a>> {
    if stream.eat(Kind::LParen) {
        let inner = stream.nested(parse_condition)?;
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
    Ok(Some((op, expr::parse_arithmetic(stream)?)))
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
    let target = expr::parse(stream)?;
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
    Ok(StmtKind::Switch { target, arms })
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
    if let Some(line) = end_line(&body) {
        return Err(PrismError::Parse {
            line,
            message: format!("`end` inside the body of gate `{name}`"),
        });
    }
    Ok(StmtKind::GateDef {
        name,
        params,
        qubits,
        body,
    })
}

/// The line of the first `end` anywhere in `body`, searched without recursion.
fn end_line(body: &Block) -> Option<usize> {
    let mut pending = vec![body];
    while let Some(block) = pending.pop() {
        for stmt in block {
            match &stmt.kind {
                StmtKind::End => return Some(stmt.line),
                StmtKind::If(conditional) => {
                    pending.push(&conditional.then_body);
                    pending.extend(&conditional.else_body);
                }
                StmtKind::For { body, .. }
                | StmtKind::While { body, .. }
                | StmtKind::Box { body, .. }
                | StmtKind::DefDef { body, .. }
                | StmtKind::GateDef { body, .. } => pending.push(body),
                StmtKind::Switch { arms, .. } => pending.extend(arms.iter().map(|arm| &arm.body)),
                _ => {}
            }
        }
    }
    None
}

fn def_def<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    let line = stream.line();
    stream.advance();
    let name = stream.expect_ident()?;
    stream.expect(Kind::LParen)?;
    let mut args = Vec::new();
    while !stream.eat(Kind::RParen) {
        if stream.is_keyword("readonly") || stream.is_keyword("mutable") {
            args.push(array_parameter(stream)?);
            if !stream.eat(Kind::Comma) {
                stream.expect(Kind::RParen)?;
                break;
            }
            continue;
        }
        let (ty, width) = typename(stream)?;
        let arg_name = stream.expect_ident()?;
        args.push(match ty {
            "qubit" => DefParam::Qubit {
                name: arg_name,
                width,
            },
            "int" | "uint" | "bool" | "float" | "angle" | "complex" => DefParam::Value {
                name: arg_name,
                ty,
                width,
            },
            "duration" | "stretch" => DefParam::Duration(arg_name),
            "bit" => DefParam::Bit {
                name: arg_name,
                width,
            },
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
    let result = if stream.eat(Kind::Arrow) {
        let (ty, width) = typename(stream)?;
        if !matches!(
            ty,
            "bit" | "int" | "uint" | "bool" | "float" | "angle" | "complex" | "duration"
        ) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("def `{name}` returning `{ty}`"),
                line,
            });
        }
        Some(DefType { ty, width })
    } else {
        None
    };
    let body = braced_block(stream)?;
    Ok(StmtKind::DefDef {
        name,
        args,
        result,
        body,
    })
}

/// `readonly array[int[8], 2, 3] a` or `mutable array[int[8], #dim = 2] a`.
fn array_parameter<'a>(stream: &mut Stream<'_, 'a>) -> Result<DefParam<'a>> {
    let mutable = stream.advance().text == "mutable";
    if !stream.eat_keyword("array") {
        return Err(stream.expected("`array` after `readonly` or `mutable`"));
    }
    stream.expect(Kind::LBracket)?;
    let (ty, width) = typename(stream)?;
    stream.expect(Kind::Comma)?;
    let shape = if stream.eat(Kind::Dim) {
        stream.expect(Kind::Assign)?;
        ArrayShape::Rank(expr::parse(stream)?)
    } else {
        let mut dims = vec![expr::parse(stream)?];
        while stream.eat(Kind::Comma) {
            dims.push(expr::parse(stream)?);
        }
        ArrayShape::Sized(dims)
    };
    stream.expect(Kind::RBracket)?;
    let name = stream.expect_ident()?;
    Ok(DefParam::Array {
        name,
        mutable,
        ty,
        width,
        shape,
    })
}

fn return_statement<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    stream.advance();
    let value = if stream.kind() == Kind::Semicolon {
        None
    } else if stream.eat_keyword("measure") {
        Some(ReturnValue::Measure(operand(stream)?))
    } else {
        Some(ReturnValue::Value(expr::parse(stream)?))
    };
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Return(value))
}

/// A gate application, a `def` call, or an assignment, which all open with a
/// name. Only what follows the name tells them apart.
fn call_or_assignment<'a>(stream: &mut Stream<'_, 'a>) -> Result<StmtKind<'a>> {
    // A name followed by an operand, an argument list or `@` opens a call, never
    // an assignment, so the trial parse only runs where one could follow.
    let call_shape = matches!(
        stream.peek_at(1).kind,
        Kind::Ident | Kind::Physical | Kind::LParen | Kind::At
    );
    if !call_shape && let Some(kind) = try_assignment(stream)? {
        return Ok(kind);
    }
    let modifiers = modifier_chain(stream)?;
    let name = stream.expect_ident()?;
    let params = if stream.eat(Kind::LParen) {
        call_arguments(stream)?
    } else {
        SmallVec::new()
    };
    // `x[100ns] q;` asks for a gate length, which an ideal simulation has no
    // clock to honour, so the designator is read and dropped.
    if stream.eat(Kind::LBracket) {
        expr::parse(stream)?;
        stream.expect(Kind::RBracket)?;
    }
    let mut operands = SmallVec::new();
    if stream.kind() != Kind::Semicolon {
        operands.push(operand(stream)?);
        while stream.eat(Kind::Comma) {
            operands.push(operand(stream)?);
        }
    }
    stream.expect(Kind::Semicolon)?;
    Ok(StmtKind::Call {
        modifiers,
        name,
        params,
        operands,
    })
}

/// The entries of a call's parentheses, the opening `(` already read.
fn call_arguments<'a>(stream: &mut Stream<'_, 'a>) -> Result<SmallVec<[Argument<'a>; 1]>> {
    let mut args = SmallVec::new();
    while !stream.eat(Kind::RParen) {
        args.push(argument(stream)?);
        if !stream.eat(Kind::Comma) {
            stream.expect(Kind::RParen)?;
            break;
        }
    }
    Ok(args)
}

/// `c = measure q;`, `n = n + 1;` and `n += 1;`, told from a gate call by the
/// assignment that follows the target.
fn try_assignment<'a>(stream: &mut Stream<'_, 'a>) -> Result<Option<StmtKind<'a>>> {
    let mark = stream.mark();
    let Ok(target) = operand(stream) else {
        stream.rewind(mark);
        return element_assignment(stream);
    };
    let Some(op) = assign_op(stream.kind()) else {
        stream.rewind(mark);
        return element_assignment(stream);
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

    // A call standing alone on the right may hand over qubits and bits, which
    // no expression can hold, so it keeps the call's own argument grammar.
    if op.is_none()
        && stream.kind() == Kind::Ident
        && stream.peek_at(1).kind == Kind::LParen
        && expr::CastType::named(stream.peek().text).is_none()
    {
        let value_mark = stream.mark();
        let name = stream.advance().text;
        stream.advance();
        if let Ok(args) = call_arguments(stream)
            && stream.eat(Kind::Semicolon)
        {
            return Ok(Some(StmtKind::CallAssign(Box::new(CallAssign {
                target,
                name,
                args,
            }))));
        }
        stream.rewind(value_mark);
    }

    let value = value(stream)?;
    stream.expect(Kind::Semicolon)?;
    let index = match &target.index {
        Some(Index::Single(index)) => Some(index.clone()),
        Some(Index::Range(range)) => Some(Expr::Range(range.clone())),
        _ => None,
    };
    if let (Some(array), Some(index)) = (target.register(), index) {
        return Ok(Some(StmtKind::ElementAssign(Box::new(ElementAssign {
            array,
            indices: vec![index],
            op,
            value,
        }))));
    }
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

/// The assignment an operator token spells: `None` when it is no assignment,
/// `Some(None)` for a plain `=`.
fn assign_op(kind: Kind) -> Option<Option<AssignOp>> {
    match kind {
        Kind::Assign => Some(None),
        Kind::AddAssign => Some(Some(AssignOp::Add)),
        Kind::SubAssign => Some(Some(AssignOp::Sub)),
        Kind::MulAssign => Some(Some(AssignOp::Mul)),
        Kind::DivAssign => Some(Some(AssignOp::Div)),
        Kind::ModAssign => Some(Some(AssignOp::Rem)),
        Kind::PowAssign => Some(Some(AssignOp::Pow)),
        Kind::AndAssign => Some(Some(AssignOp::BitAnd)),
        Kind::OrAssign => Some(Some(AssignOp::BitOr)),
        Kind::XorAssign => Some(Some(AssignOp::BitXor)),
        Kind::ShlAssign => Some(Some(AssignOp::Shl)),
        Kind::ShrAssign => Some(Some(AssignOp::Shr)),
        _ => None,
    }
}

/// `a[0, 2] = 1;` and `a[0][2] += 1;`, the element forms no operand spells.
fn element_assignment<'a>(stream: &mut Stream<'_, 'a>) -> Result<Option<StmtKind<'a>>> {
    if stream.kind() != Kind::Ident || stream.peek_at(1).kind != Kind::LBracket {
        return Ok(None);
    }
    let mark = stream.mark();
    let array = stream.advance().text;
    let indices = expr::index_groups(stream);
    let (Ok(indices), Some(op)) = (indices, assign_op(stream.kind())) else {
        stream.rewind(mark);
        return Ok(None);
    };
    stream.advance();
    let value = value(stream)?;
    stream.expect(Kind::Semicolon)?;
    Ok(Some(StmtKind::ElementAssign(Box::new(ElementAssign {
        array,
        indices,
        op,
        value,
    }))))
}

/// `inv @`, `pow(k) @`, `ctrl @`, `negctrl @` and their counted forms
/// `ctrl(n) @` and `negctrl(n) @`, in any order and chainable.
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
        let mut argument = None;
        if stream.kind() == Kind::LParen {
            stream.advance();
            let parsed = expr::parse(stream);
            match parsed {
                Ok(value) if stream.eat(Kind::RParen) => argument = Some(value),
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
        out.push(match (word, argument) {
            ("inv", None) => Modifier::Inv,
            ("ctrl", count) => Modifier::Ctrl {
                negated: false,
                count,
            },
            ("negctrl", count) => Modifier::Ctrl {
                negated: true,
                count,
            },
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

/// One entry inside a call's parentheses. A subscripted name standing alone is
/// a qubit or bit reference, or an array element, which only the declaration
/// tells apart; anything else parses as a value, and a bare name is read as
/// whichever the declaration asks for.
fn argument<'a>(stream: &mut Stream<'_, 'a>) -> Result<Argument<'a>> {
    if stream.kind() == Kind::Physical
        || (stream.kind() == Kind::Ident && stream.peek_at(1).kind == Kind::LBracket)
    {
        let mark = stream.mark();
        if let Ok(operand) = operand(stream)
            && matches!(stream.kind(), Kind::Comma | Kind::RParen)
        {
            return Ok(Argument::Operand(operand));
        }
        stream.rewind(mark);
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
