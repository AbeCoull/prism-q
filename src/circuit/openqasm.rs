//! OpenQASM 3.0 parser, v0 subset.
//!
//! # Supported constructs
//!
//! | Construct | Example | Notes |
//! |-----------|---------|-------|
//! | Header | `OPENQASM 3.0;` | 2.0 also accepted for compat |
//! | Include | `include "stdgates.inc";` | Accepted, ignored (gates built-in) |
//! | Qubit declaration | `qubit[4] q;` | OQ3 syntax (primary) |
//! | Bit declaration | `bit[4] c;` | OQ3 syntax (primary) |
//! | Legacy qreg/creg | `qreg q[4]; creg c[4];` | OQ2 compat |
//! | Input parameter | `input float[64] theta;` | One named slot; [`parse_parametric`] returns them |
//! | Output declaration | `output bit[4] c;` | Declares the register; every bit is reported anyway |
//! | 1-qubit gates | `h q[0]; x q[1];` | id, x, y, z, h, s, sdg, t, tdg, sx, sxdg, p/phase, r, gpi, gpi2, u/U forms |
//! | Parametric gates | `rx(pi/4) q[0];` | rx, ry, rz, cu, ms, arithmetic expressions with `pi`, math functions |
//! | 2-qubit gates | `cx q[0], q[1];` | cx/cnot, cy, cz, ch, cs, csdg, cp/cphase, crx, cry, crz, csx, swap, rzz, rxx, ryy, xx_plus_yy, xx_minus_yy, ecr, iswap, dcx, syc, sqrt_iswap |
//! | Multi-qubit gates | `ccx q[0], q[1], q[2];` | ccx/toffoli, ccz, cswap/fredkin, c3x, c4x, mcx, rccx, rc3x/rcccx |
//! | Gate modifiers | `inv @ h q[0];` | `inv @`, `ctrl @` (chainable), `pow(k) @` (integer k) for direct gates |
//! | Measurement (OQ3) | `c[0] = measure q[0];` | Assignment syntax (primary) |
//! | Measurement (OQ2) | `measure q[0] -> c[0];` | Arrow syntax (compat) |
//! | Register broadcast | `h q;` / `cx q, r;` | Applies gate to all qubits in register |
//! | Conditional (OQ2) | `if(c==1) x q[0];` | Classical register equality |
//! | Conditional (OQ3) | `if (c[0]) x q[0];` | Single classical bit test |
//! | Conditional inequality | `if (c != 0) x q[0];` | Register or bit `!=` |
//! | Conditional bit literal | `if (c[0] == 1) x q[0];` | Bit equality vs `0` / `1` |
//! | Conditional negation | `if (!c[0]) x q[0];` | Negated bit truthy test |
//! | Guarded region | `if (c[0]) { x q[0]; measure q[1] -> c[1]; }` | Braced body, any statement, nestable |
//! | Conditional parity | `if (c[0] ^ c[2]) x q[0];` | Parity over bits, optionally `(...) == 0` |
//! | Else arm | `if (c[0]) { ... } else { ... }` | Lowers to a second guard on the negated condition |
//! | Else-if chain | `if (c[0]) { ... } else if (c[1]) { ... }` | Nests under the negated arm |
//! | Switch | `switch (c) { case 0 { ... } default { ... } }` | Lowers to one guard per case label |
//! | Hex / binary literals | `if (c == 0xff) ...` | `0x`, `0b`, `0o` integer prefixes with optional `_` separators |
//! | Boolean literals | `rx(true * pi) ...` | `true` / `false` evaluate to `1.0` / `0.0` |
//! | Gate definition | `gate rxx(t) a,b { ... }` | User-defined gates |
//! | Subroutine definition | `def myg(qubit a, float t) { ... }` | Unitary `def` bodies, inlined at the call site |
//! | Static for loop | `for int i in [0:n] { ... }` | Inclusive ranges, optional step, set form `{a,b,c}` |
//! | Barrier | `barrier q[0], q[1];` | |
//! | Line comments | `// comment` | |
//!
//! # Unsupported constructs (return `PrismError::UnsupportedConstruct`)
//!
//! - `defcal`, `extern`, `opaque`, `box`, `while`, `return`, `break`
//! - `def` bodies that contain `measure`, `reset`, `bit`, `creg`, `return`,
//!   or the `=measure` assignment shape (V1 supports unitary subroutines only)
//! - `def` declarations with a return type
//! - `ctrl @ swap` modifier form (use `cswap` or `fredkin` keyword instead)
//! - `pow(k) @` with non-integer k (fractional powers)
//! - Bit literal comparisons against integers other than `0` / `1`
//! - Negative integer literals in `if` register comparisons
//! - `else` whose `if` body measures into a bit the condition reads, and
//!   `switch` whose arm measures into the switched register: both lowerings
//!   re-read the bits after an earlier body ran
//! - `switch` with a `default` and more case labels than the region depth bound
//! - `duration`, `stretch` outside `def` parameter lists
//! - `input` of any type but `float` and `angle`, and `output` of any type but
//!   `bit`
//! - an `input` anywhere but as the whole angle argument of a top-level
//!   parametric gate: an expression over one, two on one gate, a modified gate,
//!   a gate carrying no rotation angle, and a use inside a `gate`, `def`,
//!   `for`, or guarded-region body all reject
//!
//! # Error behaviour
//!
//! All parse failures return `PrismError::Parse` or `PrismError::UnsupportedConstruct`
//! with the source line number. The parser never panics on user input.
//!
//! The reverse direction is [`qasm_export`](super::qasm_export).

use num_complex::Complex64;

use crate::circuit::{
    Circuit, ClassicalCondition, Instruction, MAX_REGION_DEPTH, ParamLink, Parameters, SmallVec,
    guarded, smallvec,
};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use std::collections::HashMap;

/// Parse an OpenQASM 3.0 string into a PRISM-Q [`Circuit`].
///
/// This is the primary input entrypoint. The entire parse happens in-memory
/// from the provided `&str`, no file I/O.
///
/// # Errors
///
/// Returns structured [`PrismError`] for any parse failure or unsupported
/// construct, and [`PrismError::InvalidParameter`] when the program declares an
/// `input`, whose value this entry point has nowhere to take. Use
/// [`parse_parametric`] for those.
pub fn parse(input: &str) -> Result<Circuit> {
    let (circuit, params) = Parser::new(input).parse()?;
    if params.num_slots() > 0 {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "program declares {} `input` parameter(s); parse it with `parse_parametric` and bind them",
                params.num_slots()
            ),
        });
    }
    Ok(circuit)
}

/// Parse an OpenQASM 3.0 string into a [`Circuit`] plus the [`Parameters`] its
/// `input` declarations name.
///
/// Slots are ordered by declaration and carry the declared names, so
/// [`Parameters::slot_of`] resolves an angle by the name the source used. The
/// returned circuit is a template holding zero for every input; bind it before
/// running, since dispatch reads the angles it is given.
///
/// An `input` may only be the whole angle argument of a directly named
/// parametric gate at the top level. An expression over one, a use inside a
/// gate, `def`, `for`, or guarded-region body, and a use on a gate carrying no
/// rotation angle all return [`PrismError::UnsupportedConstruct`].
///
/// # Errors
///
/// Same conditions as [`parse`], less the `input` rejection.
pub fn parse_parametric(input: &str) -> Result<(Circuit, Parameters)> {
    Parser::new(input).parse()
}

enum Modifier {
    Inv,
    Ctrl,
    Pow(i64),
}

struct Register {
    offset: usize,
    size: usize,
}

struct GateDefinition {
    params: Vec<String>,
    qubits: Vec<String>,
    body: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DefParamKind {
    Float,
    Int,
}

enum DefArg {
    Qubit(String),
    Param { name: String, kind: DefParamKind },
}

struct DefDefinition {
    args: Vec<DefArg>,
    body: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Gate,
    Def,
    For,
    If,
    Switch,
}

#[derive(Clone, Copy)]
enum RegisterKind {
    Qubit,
    Classical,
}

impl RegisterKind {
    fn name(self) -> &'static str {
        match self {
            RegisterKind::Qubit => "qubit",
            RegisterKind::Classical => "bit",
        }
    }

    fn invalid_index(self, index: usize, register_size: usize) -> PrismError {
        match self {
            RegisterKind::Qubit => PrismError::InvalidQubit {
                index,
                register_size,
            },
            RegisterKind::Classical => PrismError::InvalidClassicalBit {
                index,
                register_size,
            },
        }
    }
}

struct BlockState {
    kind: BlockKind,
    buf: String,
    start_line: usize,
    depth: usize,
}

pub(crate) struct Parser<'a> {
    input: &'a str,
    qregs: HashMap<String, Register>,
    cregs: HashMap<String, Register>,
    gate_defs: HashMap<String, GateDefinition>,
    def_defs: HashMap<String, DefDefinition>,
    total_qubits: usize,
    total_cbits: usize,
    gate_expansion_depth: usize,
    region_depth: usize,
    param_vars: Option<HashMap<String, f64>>,
    int_vars: Option<HashMap<String, i64>>,
    /// `input` slot per declared name, and the names in slot order.
    inputs: HashMap<String, usize>,
    input_names: Vec<String>,
    links: Vec<ParamLink>,
    /// Slot the statement just parsed reads, handed from `process_top_line` up
    /// to `parse_lines`, which is the only place that knows the instruction
    /// index the link needs.
    pending_input_slot: Option<usize>,
    /// True while parsing a block body, whose instruction indices are local to
    /// that body and so cannot carry a top-level parameter link.
    nested: bool,
}

const MAX_GATE_EXPANSION_DEPTH: usize = 32;
const MAX_FOR_ITERATIONS: i64 = 1_000_000;

use super::expr::{contains_word, eval_expr, replace_word, split_top_level_commas};

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(pos) => &line[..pos],
        None => line,
    }
}

fn block_kind_name(kind: BlockKind) -> &'static str {
    match kind {
        BlockKind::Gate => "gate",
        BlockKind::Def => "def",
        BlockKind::For => "for",
        BlockKind::If => "if",
        BlockKind::Switch => "switch",
    }
}

fn update_brace_depth(mut depth: usize, line: &str) -> usize {
    for ch in line.chars() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    depth
}

fn extract_top_braced_body(s: &str) -> Option<(usize, usize)> {
    let open = s.find('{')?;
    let mut depth = 0usize;
    for (i, ch) in s[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((open, open + i));
                }
            }
            _ => {}
        }
    }
    None
}

fn find_matching_close_paren(s: &str) -> Option<usize> {
    let mut depth = 1usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// True when the next statement after a closed `if` block is its `else`.
///
/// An `else` on its own line would otherwise reach the top-level dispatcher,
/// which rejects the keyword by name.
fn awaits_else(kind: BlockKind, lines: &[&str], line_idx: usize) -> bool {
    if kind != BlockKind::If {
        return false;
    }
    lines[line_idx + 1..]
        .iter()
        .map(|line| strip_comment(line).trim())
        .find(|line| !line.is_empty())
        .is_some_and(|line| strip_leading_keyword(line, "else").is_some())
}

/// True for an unbraced `if` whose `else` sits on a later line, which has to
/// buffer so both arms reach the same parse.
fn opens_split_else(line: &str, lines: &[&str], line_idx: usize) -> bool {
    strip_leading_keyword(line, "if").is_some()
        && !line.contains('{')
        && awaits_else(BlockKind::If, lines, line_idx)
}

/// Split an unbraced `if` body at its `else`, or return the whole body when it
/// has none.
fn split_at_else(s: &str) -> (&str, &str) {
    let mut depth = 0usize;
    for (index, _) in s.char_indices() {
        match s.as_bytes()[index] {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b'e' if depth == 0 && strip_leading_keyword(&s[index..], "else").is_some() => {
                let preceded_by_word = s[..index]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
                if !preceded_by_word {
                    return (&s[..index], &s[index..]);
                }
            }
            _ => {}
        }
    }
    (s, "")
}

/// True when a buffered block ends on the `else` keyword, so its arm is still
/// on a later line.
fn ends_on_bare_else(buf: &str) -> bool {
    let trimmed = buf.trim_end();
    let Some(head) = trimmed.strip_suffix("else") else {
        return false;
    };
    !head.ends_with(|c: char| c.is_alphanumeric() || c == '_')
}

/// Strip `keyword` when it opens `s` as a whole word, returning the rest.
fn strip_leading_keyword<'a>(s: &'a str, keyword: &str) -> Option<&'a str> {
    let rest = s.trim_start().strip_prefix(keyword)?;
    if rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
        return None;
    }
    Some(rest.trim_start())
}

/// Split a leading `{ ... }` off `s`, returning the body and what follows it.
fn split_braced_body<'a>(s: &'a str, line_num: usize, what: &str) -> Result<(&'a str, &'a str)> {
    let s = s.trim_start();
    let open = s.find('{').ok_or_else(|| PrismError::Parse {
        line: line_num,
        message: format!("expected `{{` in `{what}` body"),
    })?;
    if !s[..open].trim().is_empty() {
        return Err(PrismError::Parse {
            line: line_num,
            message: format!("unexpected `{}` before `{what}` body", s[..open].trim()),
        });
    }
    let after = &s[open + 1..];
    let close = find_matching_close_brace(after).ok_or_else(|| PrismError::Parse {
        line: line_num,
        message: format!("unmatched `{{` in `{what}` body"),
    })?;
    Ok((&after[..close], &after[close + 1..]))
}

/// One `case` or `default` arm of a `switch`; `labels_src` is `None` for
/// `default`.
struct SwitchArm<'a> {
    labels_src: Option<&'a str>,
    body: &'a str,
}

fn split_switch_arms<'a>(src: &'a str, line_num: usize) -> Result<Vec<SwitchArm<'a>>> {
    let mut out = Vec::new();
    let mut rest = src.trim();
    while !rest.is_empty() {
        if let Some(after) = strip_leading_keyword(rest, "case") {
            let open = after.find('{').ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "expected `{` after `switch` case labels".to_string(),
            })?;
            let (body, tail) = split_braced_body(&after[open..], line_num, "case")?;
            out.push(SwitchArm {
                labels_src: Some(&after[..open]),
                body,
            });
            rest = tail.trim();
        } else if let Some(after) = strip_leading_keyword(rest, "default") {
            let (body, tail) = split_braced_body(after, line_num, "default")?;
            out.push(SwitchArm {
                labels_src: None,
                body,
            });
            rest = tail.trim();
        } else {
            let word = rest
                .split(|c: char| c.is_whitespace() || c == '(' || c == '{')
                .next()
                .unwrap_or(rest);
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("expected `case` or `default` in `switch` body, got `{word}`"),
            });
        }
    }
    Ok(out)
}

/// Wrap `body` in one negated-equality region per case label, which is how a
/// conjunction is spelled in a condition language that has no `and`.
fn nest_default_arm(
    offset: usize,
    size: usize,
    labels: &[u64],
    body: Vec<Instruction>,
    region_depth: usize,
    line_num: usize,
) -> Result<Vec<Instruction>> {
    // The nesting the default costs is on top of wherever the `switch` sits, so
    // the bound is the sum rather than the label count alone.
    if region_depth + labels.len() > MAX_REGION_DEPTH {
        return Err(PrismError::Parse {
            line: line_num,
            message: format!(
                "`switch` with a `default` nests one region per case label and would \
                 pass the depth bound of {MAX_REGION_DEPTH} at {} labels",
                labels.len()
            ),
        });
    }
    let mut current = body;
    for &value in labels.iter().rev() {
        let condition = ClassicalCondition::RegisterNotEquals {
            offset,
            size,
            value,
        };
        current = guarded(condition, current).into_iter().collect();
    }
    Ok(current)
}

fn find_matching_close_brace(s: &str) -> Option<usize> {
    let mut depth = 1usize;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn is_ident_char_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn find_keyword(haystack: &str, needle: &str) -> Option<usize> {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    let nlen = nb.len();
    let mut i = 0;
    while i + nlen <= hb.len() {
        if &hb[i..i + nlen] == nb {
            let before_ok = i == 0 || !is_ident_char_byte(hb[i - 1]);
            let after_ok = i + nlen >= hb.len() || !is_ident_char_byte(hb[i + nlen]);
            if before_ok && after_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn parse_for_var(lhs: &str, line_num: usize) -> Result<String> {
    let mut tokens = lhs.split_whitespace();
    let first = tokens.next().ok_or_else(|| PrismError::Parse {
        line: line_num,
        message: "missing loop variable in for header".to_string(),
    })?;

    let var = if matches!(first, "int" | "uint") {
        let next = tokens.next().ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "missing loop variable name after type in for header".to_string(),
        })?;
        next.trim_end_matches(',').to_string()
    } else if first.starts_with("int[") || first.starts_with("uint[") {
        let next = tokens.next().ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "missing loop variable name after type in for header".to_string(),
        })?;
        next.trim_end_matches(',').to_string()
    } else {
        first.to_string()
    };

    if tokens.next().is_some() {
        return Err(PrismError::Parse {
            line: line_num,
            message: format!("unexpected tokens in for loop variable spec: `{lhs}`"),
        });
    }

    if var.is_empty()
        || !var.chars().next().unwrap().is_ascii_alphabetic()
        || !var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(PrismError::Parse {
            line: line_num,
            message: format!("invalid loop variable name: `{var}`"),
        });
    }

    Ok(var)
}

fn eval_int_expr(s: &str, line_num: usize, vars: Option<&HashMap<String, i64>>) -> Result<i64> {
    let float_vars: Option<HashMap<String, f64>> = vars.map(|m| {
        m.iter()
            .map(|(k, v)| (k.clone(), *v as f64))
            .collect::<HashMap<_, _>>()
    });
    let val = eval_expr(s, line_num, float_vars.as_ref())?;
    if val.fract() != 0.0 || !val.is_finite() {
        return Err(PrismError::Parse {
            line: line_num,
            message: format!("expected integer expression, got `{s}` = {val}"),
        });
    }
    if val > i64::MAX as f64 || val < i64::MIN as f64 {
        return Err(PrismError::Parse {
            line: line_num,
            message: format!("integer expression `{s}` out of range"),
        });
    }
    Ok(val as i64)
}

fn parse_for_range(
    rhs: &str,
    line_num: usize,
    int_vars: Option<&HashMap<String, i64>>,
) -> Result<Vec<i64>> {
    let rhs = rhs.trim();
    if let Some(inner) = rhs.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let parts: Vec<&str> = inner.split(':').collect();
        let (start, step, stop) = match parts.len() {
            2 => (
                eval_int_expr(parts[0].trim(), line_num, int_vars)?,
                1i64,
                eval_int_expr(parts[1].trim(), line_num, int_vars)?,
            ),
            3 => (
                eval_int_expr(parts[0].trim(), line_num, int_vars)?,
                eval_int_expr(parts[1].trim(), line_num, int_vars)?,
                eval_int_expr(parts[2].trim(), line_num, int_vars)?,
            ),
            _ => {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!("malformed range `[{inner}]` in for loop"),
                });
            }
        };
        if step == 0 {
            return Err(PrismError::Parse {
                line: line_num,
                message: "for loop range step must be non-zero".to_string(),
            });
        }
        let mut values = Vec::new();
        let mut i = start;
        if step > 0 {
            while i <= stop {
                values.push(i);
                if values.len() as i64 > MAX_FOR_ITERATIONS {
                    return Err(PrismError::Parse {
                        line: line_num,
                        message: format!("for loop iterates more than {MAX_FOR_ITERATIONS} times"),
                    });
                }
                i += step;
            }
        } else {
            while i >= stop {
                values.push(i);
                if values.len() as i64 > MAX_FOR_ITERATIONS {
                    return Err(PrismError::Parse {
                        line: line_num,
                        message: format!("for loop iterates more than {MAX_FOR_ITERATIONS} times"),
                    });
                }
                i += step;
            }
        }
        return Ok(values);
    }
    if let Some(inner) = rhs.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        let mut values = Vec::new();
        for raw in split_top_level_commas(inner) {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }
            values.push(eval_int_expr(token, line_num, int_vars)?);
        }
        return Ok(values);
    }
    Err(PrismError::UnsupportedConstruct {
        construct: format!(
            "for loop range `{rhs}` (only `[start:stop]`, `[start:step:stop]`, or `{{a,b,c}}` supported)"
        ),
        line: line_num,
    })
}

fn split_body_into_lines(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for ch in body.chars() {
        match ch {
            '{' => {
                depth += 1;
                current.push(ch);
            }
            '}' => {
                depth = depth.saturating_sub(1);
                current.push(ch);
                if depth == 0 {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        out.push(trimmed);
                    }
                    current.clear();
                }
            }
            ';' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    out.push(trimmed);
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        out.push(trimmed);
    }
    out
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            gate_defs: HashMap::new(),
            def_defs: HashMap::new(),
            total_qubits: 0,
            total_cbits: 0,
            gate_expansion_depth: 0,
            region_depth: 0,
            param_vars: None,
            int_vars: None,
            inputs: HashMap::new(),
            input_names: Vec::new(),
            links: Vec::new(),
            pending_input_slot: None,
            nested: false,
        }
    }

    fn parse(mut self) -> Result<(Circuit, Parameters)> {
        let lines: Vec<&str> = self.input.lines().collect();
        let instructions = self.parse_lines(&lines, 0)?;

        let circuit = Circuit {
            num_qubits: self.total_qubits,
            num_classical_bits: self.total_cbits,
            instructions,
        };
        let params = Parameters::from_links(self.links, self.input_names.len())
            .with_names(self.input_names)
            .pinned_to(&circuit);
        Ok((circuit, params))
    }

    fn parse_lines(&mut self, lines: &[&str], line_offset: usize) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();
        let mut block: Option<BlockState> = None;

        for (line_idx, raw_line) in lines.iter().enumerate() {
            let line_num = line_offset + line_idx + 1;

            let line = strip_comment(raw_line).trim();
            if line.is_empty() {
                continue;
            }

            if let Some(state) = block.as_mut() {
                state.buf.push(' ');
                state.buf.push_str(line);
                state.depth = update_brace_depth(state.depth, line);
            } else if opens_split_else(line, lines, line_idx) {
                block = Some(BlockState {
                    kind: BlockKind::If,
                    buf: line.to_string(),
                    start_line: line_num,
                    depth: 0,
                });
            } else {
                let base = instructions.len();
                let produced = self.process_top_line(line, line_num, &mut block)?;
                if let Some(slot) = self.pending_input_slot.take() {
                    for offset in 0..produced.len() {
                        self.links.push(ParamLink {
                            instruction: base + offset,
                            slot,
                        });
                    }
                }
                instructions.extend(produced);
            }

            if let Some(state) = block.as_ref()
                && state.depth == 0
                && !awaits_else(state.kind, lines, line_idx)
                && !ends_on_bare_else(&state.buf)
            {
                let finished = block.take().unwrap();
                instructions.extend(self.dispatch_block(&finished)?);
            }
        }

        if let Some(state) = block {
            return Err(PrismError::Parse {
                line: state.start_line,
                message: format!(
                    "unterminated `{}` block (missing `}}`)",
                    block_kind_name(state.kind)
                ),
            });
        }

        Ok(instructions)
    }

    fn process_top_line(
        &mut self,
        line: &str,
        line_num: usize,
        block: &mut Option<BlockState>,
    ) -> Result<Vec<Instruction>> {
        self.pending_input_slot = None;
        let first_word = line
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or(line);

        // A braced `if` is a guarded region and takes the block path; the
        // one-statement `if (c) x q[0];` form has no brace and falls through to
        // `parse_if_statement` below. The caller dispatches the block once its
        // braces balance and no `else` follows.
        if matches!(first_word, "gate" | "def" | "for" | "switch")
            || (first_word == "if" && line.contains('{'))
        {
            let kind = match first_word {
                "gate" => BlockKind::Gate,
                "def" => BlockKind::Def,
                "for" => BlockKind::For,
                "if" => BlockKind::If,
                "switch" => BlockKind::Switch,
                _ => unreachable!(),
            };
            if !line.contains('{') {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!("expected `{{` in `{}` block", first_word),
                });
            }
            *block = Some(BlockState {
                kind,
                buf: line.to_string(),
                start_line: line_num,
                depth: update_brace_depth(0, line),
            });
            return Ok(Vec::new());
        }

        let line = line.strip_suffix(';').unwrap_or(line).trim();
        if line.is_empty() {
            return Ok(Vec::new());
        }

        if line.starts_with("OPENQASM") {
            return Ok(Vec::new());
        }
        if line.starts_with("include") {
            return Ok(Vec::new());
        }

        if line.starts_with("qubit") {
            self.parse_qubit_decl(line, line_num)?;
            return Ok(Vec::new());
        }
        if line.starts_with("bit") && !line.starts_with("bits") {
            self.parse_bit_decl(line, line_num)?;
            return Ok(Vec::new());
        }
        if first_word == "input" {
            self.parse_input_decl(line, line_num)?;
            return Ok(Vec::new());
        }
        if first_word == "output" {
            self.parse_output_decl(line, line_num)?;
            return Ok(Vec::new());
        }
        if line.starts_with("qreg") {
            self.parse_qreg_legacy(line, line_num)?;
            return Ok(Vec::new());
        }
        if line.starts_with("creg") {
            self.parse_creg_legacy(line, line_num)?;
            return Ok(Vec::new());
        }

        if line.starts_with("measure") {
            return self.parse_measure_arrow(line, line_num);
        }

        if line.contains("= measure") || line.contains("=measure") {
            return self.parse_measure_assign(line, line_num);
        }

        if line.starts_with("barrier") {
            return Ok(vec![self.parse_barrier(line, line_num)?]);
        }

        if line.starts_with("reset") {
            return self.parse_reset(line, line_num);
        }

        if line.starts_with("if") {
            if split_at_else(line).1.is_empty() {
                return self.parse_if_statement(line, line_num);
            }
            return self.parse_if_block(line, line_num);
        }

        // Recomputed: the trailing `;` is gone now, so a bare `break;` reports
        // the keyword rather than falling through to gate parsing.
        let keyword = line
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or(line);
        if matches!(
            keyword,
            "defcal" | "opaque" | "while" | "box" | "extern" | "return" | "else" | "break"
        ) {
            return Err(PrismError::UnsupportedConstruct {
                construct: keyword.to_string(),
                line: line_num,
            });
        }

        let (instrs, slot) = self.parse_gate_application(line, line_num)?;
        self.pending_input_slot = slot;
        Ok(instrs)
    }

    fn dispatch_block(&mut self, state: &BlockState) -> Result<Vec<Instruction>> {
        match state.kind {
            BlockKind::Gate => {
                self.parse_gate_def(&state.buf, state.start_line)?;
                Ok(Vec::new())
            }
            BlockKind::Def => {
                self.parse_def_block(&state.buf, state.start_line)?;
                Ok(Vec::new())
            }
            BlockKind::For => self.expand_for_block(&state.buf, state.start_line),
            BlockKind::If => self.parse_if_block(&state.buf, state.start_line),
            BlockKind::Switch => self.parse_switch_block(&state.buf, state.start_line),
        }
    }

    /// OQ3 syntax: `qubit[4] q` or `qubit q` (single qubit).
    fn parse_qubit_decl(&mut self, line: &str, line_num: usize) -> Result<()> {
        let (name, size) =
            Self::parse_oq3_register_decl(line, "qubit", RegisterKind::Qubit, line_num)?;
        let offset = self.total_qubits;
        self.total_qubits += size;
        self.qregs.insert(name, Register { offset, size });
        Ok(())
    }

    /// `input float[64] theta` declares one parameter slot, named and ordered
    /// by declaration. The template holds zero until a binding writes it.
    fn parse_input_decl(&mut self, line: &str, line_num: usize) -> Result<()> {
        let rest = line.strip_prefix("input").unwrap().trim();
        let (ty, name) = Self::split_declared_type(rest, "input", line_num)?;
        if !matches!(Self::type_keyword(ty), "float" | "angle") {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("`input {ty}` (only float and angle inputs bind to an angle)"),
                line: line_num,
            });
        }
        if self.inputs.contains_key(&name) {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("input `{name}` declared twice"),
            });
        }
        self.inputs.insert(name.clone(), self.input_names.len());
        self.input_names.push(name);
        Ok(())
    }

    /// `output bit[4] c` declares the register and marks it as the program's
    /// result. Every classical bit is already reported, so the marking is
    /// carried by the declaration alone.
    fn parse_output_decl(&mut self, line: &str, line_num: usize) -> Result<()> {
        let rest = line.strip_prefix("output").unwrap().trim();
        let (ty, _) = Self::split_declared_type(rest, "output", line_num)?;
        if Self::type_keyword(ty) != "bit" {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("`output {ty}` (only bit outputs are reported)"),
                line: line_num,
            });
        }
        self.parse_bit_decl(rest, line_num)
    }

    /// Split `float[64] theta` into its type and its name.
    fn split_declared_type<'s>(
        rest: &'s str,
        keyword: &str,
        line_num: usize,
    ) -> Result<(&'s str, String)> {
        let split = rest
            .rfind(|c: char| c.is_whitespace())
            .ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: format!("expected `{keyword} <type> <name>`, got `{keyword} {rest}`"),
            })?;
        let (ty, name) = (rest[..split].trim(), rest[split..].trim());
        if ty.is_empty() || name.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("expected `{keyword} <type> <name>`, got `{keyword} {rest}`"),
            });
        }
        Ok((ty, name.to_string()))
    }

    /// Type name with any width suffix dropped: `float[64]` reads as `float`.
    fn type_keyword(ty: &str) -> &str {
        ty.split('[').next().unwrap_or(ty).trim()
    }

    /// Evaluate one gate argument, reporting the slot when it is exactly an
    /// `input` name.
    fn resolve_param(&self, expr: &str, line_num: usize) -> Result<(f64, Option<usize>)> {
        let expr = expr.trim();
        if let Some(&slot) = self.inputs.get(expr) {
            if self.nested {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("input `{expr}` inside a block body"),
                    line: line_num,
                });
            }
            return Ok((0.0, Some(slot)));
        }
        if let Some(name) = self.input_names.iter().find(|n| contains_word(expr, n)) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "expression `{expr}` over input `{name}`; an input binds an angle whole"
                ),
                line: line_num,
            });
        }
        Ok((eval_expr(expr, line_num, self.param_vars.as_ref())?, None))
    }

    /// OQ3 syntax: `bit[4] c` or `bit c` (single bit).
    fn parse_bit_decl(&mut self, line: &str, line_num: usize) -> Result<()> {
        let (name, size) =
            Self::parse_oq3_register_decl(line, "bit", RegisterKind::Classical, line_num)?;
        let offset = self.total_cbits;
        self.total_cbits += size;
        self.cregs.insert(name, Register { offset, size });
        Ok(())
    }

    fn parse_oq3_register_decl(
        line: &str,
        keyword: &str,
        kind: RegisterKind,
        line_num: usize,
    ) -> Result<(String, usize)> {
        let rest = line.strip_prefix(keyword).unwrap();
        let kind_name = kind.name();

        if rest.trim_start().starts_with('[') {
            let bracket_content = Self::extract_bracket(rest).ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: format!("missing `]` in {kind_name} declaration"),
            })?;
            let size: usize = bracket_content.parse().map_err(|_| PrismError::Parse {
                line: line_num,
                message: format!("invalid {kind_name} count: `{bracket_content}`"),
            })?;
            if size == 0 {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!("{kind_name} count must be > 0"),
                });
            }
            let end = rest.find(']').unwrap(); // safe: extract_bracket succeeded
            let name = rest[end + 1..].trim().to_string();
            if name.is_empty() {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!("{kind_name} declaration missing name"),
                });
            }
            return Ok((name, size));
        }

        let name = rest.trim().to_string();
        if name.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("{kind_name} declaration missing name"),
            });
        }
        Ok((name, 1))
    }

    /// Extract content between first `[` and `]`, if present.
    fn extract_bracket(s: &str) -> Option<&str> {
        let start = s.find('[')?;
        let end = s.find(']')?;
        Some(s[start + 1..end].trim())
    }

    /// OQ2 compat: `qreg name[size]`.
    fn parse_qreg_legacy(&mut self, line: &str, line_num: usize) -> Result<()> {
        let rest = line.strip_prefix("qreg").unwrap().trim();
        let (name, size) = Self::parse_legacy_register_decl(rest, line_num)?;
        let offset = self.total_qubits;
        self.total_qubits += size;
        self.qregs.insert(name, Register { offset, size });
        Ok(())
    }

    /// OQ2 compat: `creg name[size]`.
    fn parse_creg_legacy(&mut self, line: &str, line_num: usize) -> Result<()> {
        let rest = line.strip_prefix("creg").unwrap().trim();
        let (name, size) = Self::parse_legacy_register_decl(rest, line_num)?;
        let offset = self.total_cbits;
        self.total_cbits += size;
        self.cregs.insert(name, Register { offset, size });
        Ok(())
    }

    fn parse_legacy_register_decl(s: &str, line_num: usize) -> Result<(String, usize)> {
        let bracket_start = s.find('[').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: format!("expected `[` in register declaration: `{s}`"),
        })?;
        let bracket_end = s.find(']').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: format!("expected `]` in register declaration: `{s}`"),
        })?;
        let name = s[..bracket_start].trim().to_string();
        if name.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: "register name is empty".to_string(),
            });
        }
        let size_str = s[bracket_start + 1..bracket_end].trim();
        let size: usize = size_str.parse().map_err(|_| PrismError::Parse {
            line: line_num,
            message: format!("invalid register size: `{size_str}`"),
        })?;
        if size == 0 {
            return Err(PrismError::Parse {
                line: line_num,
                message: "register size must be > 0".to_string(),
            });
        }
        Ok((name, size))
    }

    /// Resolve a qubit argument that may be indexed (`q[0]`) or a bare register (`q`).
    /// Returns all matching qubit indices.
    fn resolve_qubit_arg(&self, token: &str, line_num: usize) -> Result<SmallVec<[usize; 4]>> {
        self.resolve_register_arg(&self.qregs, RegisterKind::Qubit, token, line_num)
    }

    /// Resolve a classical bit argument that may be indexed (`c[0]`) or a bare register (`c`).
    fn resolve_cbit_arg(&self, token: &str, line_num: usize) -> Result<SmallVec<[usize; 4]>> {
        self.resolve_register_arg(&self.cregs, RegisterKind::Classical, token, line_num)
    }

    fn resolve_cbit(&self, token: &str, line_num: usize) -> Result<usize> {
        self.resolve_register_index(&self.cregs, RegisterKind::Classical, token, line_num)
    }

    fn resolve_register_arg(
        &self,
        registers: &HashMap<String, Register>,
        kind: RegisterKind,
        token: &str,
        line_num: usize,
    ) -> Result<SmallVec<[usize; 4]>> {
        if token.contains('[') {
            Ok(smallvec![self.resolve_register_index(
                registers, kind, token, line_num
            )?])
        } else {
            let reg = registers
                .get(token)
                .ok_or_else(|| PrismError::UndefinedRegister {
                    name: token.to_string(),
                    line: line_num,
                })?;
            Ok((0..reg.size).map(|i| reg.offset + i).collect())
        }
    }

    fn resolve_register_index(
        &self,
        registers: &HashMap<String, Register>,
        kind: RegisterKind,
        token: &str,
        line_num: usize,
    ) -> Result<usize> {
        let (name, idx) = self.parse_indexed_ref(token, line_num)?;
        let reg = registers
            .get(name)
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: name.to_string(),
                line: line_num,
            })?;
        if idx >= reg.size {
            return Err(kind.invalid_index(idx, reg.size));
        }
        Ok(reg.offset + idx)
    }

    fn parse_indexed_ref<'b>(&self, token: &'b str, line_num: usize) -> Result<(&'b str, usize)> {
        let bracket = token.find('[').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: format!("expected indexed reference (e.g. `q[0]`), got: `{token}`"),
        })?;
        let end = token.find(']').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: format!("expected `]` in reference: `{token}`"),
        })?;
        let name = token[..bracket].trim();
        let idx_str = token[bracket + 1..end].trim();
        let idx_val = eval_int_expr(idx_str, line_num, self.int_vars.as_ref()).map_err(|_| {
            PrismError::Parse {
                line: line_num,
                message: format!("invalid index in `{token}`"),
            }
        })?;
        if idx_val < 0 {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("negative index in `{token}`"),
            });
        }
        Ok((name, idx_val as usize))
    }

    /// OQ2 compat: `measure q[0] -> c[0]` or `measure q -> c` (broadcast)
    fn parse_measure_arrow(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let rest = line.strip_prefix("measure").unwrap().trim();
        let parts: Vec<&str> = rest.split("->").collect();
        if parts.len() != 2 {
            return Err(PrismError::Parse {
                line: line_num,
                message: "expected `measure qubit -> cbit`".to_string(),
            });
        }
        let qubits = self.resolve_qubit_arg(parts[0].trim(), line_num)?;
        let cbits = self.resolve_cbit_arg(parts[1].trim(), line_num)?;
        Self::build_measurements(qubits, cbits, line_num)
    }

    /// OQ3: `c[0] = measure q[0]` or `c = measure q` (broadcast)
    fn parse_measure_assign(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(PrismError::Parse {
                line: line_num,
                message: "expected `cbit = measure qubit`".to_string(),
            });
        }
        let cbit_token = parts[0].trim();
        let measure_part = parts[1].trim();
        let qubit_token = measure_part
            .strip_prefix("measure")
            .ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "expected `measure` after `=`".to_string(),
            })?
            .trim();

        let cbits = self.resolve_cbit_arg(cbit_token, line_num)?;
        let qubits = self.resolve_qubit_arg(qubit_token, line_num)?;
        Self::build_measurements(qubits, cbits, line_num)
    }

    fn build_measurements(
        qubits: SmallVec<[usize; 4]>,
        cbits: SmallVec<[usize; 4]>,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        if qubits.len() != cbits.len() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!(
                    "register size mismatch in measure: {} qubits vs {} classical bits",
                    qubits.len(),
                    cbits.len()
                ),
            });
        }
        Ok(qubits
            .into_iter()
            .zip(cbits)
            .map(|(qubit, classical_bit)| Instruction::Measure {
                qubit,
                classical_bit,
            })
            .collect())
    }

    /// Parse a `gate name(params) qubits { body }` definition.
    ///
    /// The full definition (possibly collected from multiple lines) is in `line`.
    fn parse_gate_def(&mut self, line: &str, line_num: usize) -> Result<()> {
        let rest = line.strip_prefix("gate").unwrap().trim();

        let brace_open = rest.find('{').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `{` in gate definition".to_string(),
        })?;
        let brace_close = rest.rfind('}').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `}` in gate definition".to_string(),
        })?;

        let header = rest[..brace_open].trim();
        let body_str = rest[brace_open + 1..brace_close].trim();

        let (name, params, qubit_names) = if let Some(paren_open) = header.find('(') {
            let paren_close = header.find(')').ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "expected `)` in gate parameters".to_string(),
            })?;
            let name = header[..paren_open].trim().to_string();
            let params: Vec<String> = header[paren_open + 1..paren_close]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let qubit_names: Vec<String> = header[paren_close + 1..]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            (name, params, qubit_names)
        } else {
            let parts: Vec<&str> = header.split_whitespace().collect();
            let Some(name_tok) = parts.first() else {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: "gate definition is missing a name".to_string(),
                });
            };
            let name = name_tok.to_string();
            let qubit_names: Vec<String> = parts[1..]
                .iter()
                .flat_map(|s| s.split(','))
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            (name, Vec::new(), qubit_names)
        };

        let body: Vec<String> = body_str
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if body.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("gate '{}' has an empty body", name),
            });
        }

        self.gate_defs.insert(
            name,
            GateDefinition {
                params,
                qubits: qubit_names,
                body,
            },
        );
        Ok(())
    }

    /// Parse a `def name(args) { body }` subroutine definition.
    ///
    /// V1 supports unitary subroutines: parameters may be `qubit`, `int`/`uint`,
    /// `float`/`angle`. Return types and measurement, reset, classical side
    /// effects in the body are rejected.
    fn parse_def_block(&mut self, buf: &str, line_num: usize) -> Result<()> {
        let rest = buf.trim_start().strip_prefix("def").unwrap().trim();

        let (open, close) = extract_top_braced_body(rest).ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `{ ... }` body in def".to_string(),
        })?;
        let header = rest[..open].trim();
        let body_str = rest[open + 1..close].trim();

        if header.contains("->") {
            return Err(PrismError::UnsupportedConstruct {
                construct: "def with return type".to_string(),
                line: line_num,
            });
        }

        let paren_open = header.find('(').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `(` in def parameter list".to_string(),
        })?;
        let name = header[..paren_open].trim().to_string();
        if name.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: "missing name in def".to_string(),
            });
        }

        let after_open = &header[paren_open + 1..];
        let close_paren =
            find_matching_close_paren(after_open).ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "unmatched `(` in def parameter list".to_string(),
            })?;
        let params_str = &after_open[..close_paren];
        let trailing = after_open[close_paren + 1..].trim();
        if !trailing.is_empty() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("trailing tokens after def parameter list: `{trailing}`"),
                line: line_num,
            });
        }

        let mut args: Vec<DefArg> = Vec::new();
        for raw in split_top_level_commas(params_str) {
            let p = raw.trim();
            if p.is_empty() {
                continue;
            }
            let last_ws = p
                .rfind(char::is_whitespace)
                .ok_or_else(|| PrismError::Parse {
                    line: line_num,
                    message: format!("malformed def parameter: `{p}` (expected `<type> <name>`)"),
                })?;
            let ty = p[..last_ws].trim();
            let arg_name = p[last_ws..].trim().to_string();
            if arg_name.is_empty() {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!("missing name in def parameter: `{p}`"),
                });
            }
            let base_ty = ty.split('[').next().unwrap().trim();
            match base_ty {
                "qubit" => args.push(DefArg::Qubit(arg_name)),
                "int" | "uint" => args.push(DefArg::Param {
                    name: arg_name,
                    kind: DefParamKind::Int,
                }),
                "float" | "angle" | "complex" | "duration" | "stretch" => {
                    args.push(DefArg::Param {
                        name: arg_name,
                        kind: DefParamKind::Float,
                    })
                }
                "bit" | "creg" => {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "classical bit parameters in def `{name}` (V1 supports unitary subroutines only)"
                        ),
                        line: line_num,
                    });
                }
                other => {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("def parameter type `{other}`"),
                        line: line_num,
                    });
                }
            }
        }

        let body = split_body_into_lines(body_str);
        if body.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("def `{name}` has an empty body"),
            });
        }
        for stmt in &body {
            let first = stmt
                .split(|c: char| c.is_whitespace() || c == '(')
                .next()
                .unwrap_or("");
            match first {
                "measure" | "reset" | "return" | "bit" | "creg" => {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "`{first}` inside def `{name}` (V1 supports unitary subroutines only)"
                        ),
                        line: line_num,
                    });
                }
                _ => {}
            }
            if stmt.contains("= measure") || stmt.contains("=measure") {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("measurement inside def `{name}`"),
                    line: line_num,
                });
            }
        }

        self.def_defs.insert(name, DefDefinition { args, body });
        Ok(())
    }

    /// Expand a `for <type>? <var> in <range_or_set> { body }` loop into
    /// a sequence of instructions.
    fn expand_for_block(&mut self, buf: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let rest = buf.trim_start().strip_prefix("for").unwrap().trim();

        let in_pos = find_keyword(rest, "in").ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `in` keyword in for loop".to_string(),
        })?;
        let lhs = rest[..in_pos].trim();
        let after_in = rest[in_pos + 2..].trim_start();

        let (range_str, after_range) = if let Some(remainder) = after_in.strip_prefix('[') {
            let close = remainder.find(']').ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "expected `]` in for loop range".to_string(),
            })?;
            (
                format!("[{}]", &remainder[..close]),
                remainder[close + 1..].trim_start(),
            )
        } else if let Some(set_inner_start) = after_in.strip_prefix('{') {
            let close_offset =
                find_matching_close_brace(set_inner_start).ok_or_else(|| PrismError::Parse {
                    line: line_num,
                    message: "unmatched `{` in for loop set".to_string(),
                })?;
            (
                format!("{{{}}}", &set_inner_start[..close_offset]),
                set_inner_start[close_offset + 1..].trim_start(),
            )
        } else {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("for loop range starting with `{after_in}`"),
                line: line_num,
            });
        };

        let body_open = after_range.find('{').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `{` opening for loop body".to_string(),
        })?;
        let after_body_open = &after_range[body_open + 1..];
        let body_close =
            find_matching_close_brace(after_body_open).ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "unmatched `{` in for loop body".to_string(),
            })?;
        let body_str = after_body_open[..body_close].trim();
        let trailing = after_body_open[body_close + 1..].trim();
        if !trailing.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("unexpected tokens after for loop body: `{trailing}`"),
            });
        }

        let var_name = parse_for_var(lhs, line_num)?;
        let values = parse_for_range(&range_str, line_num, self.int_vars.as_ref())?;

        if values.len() as i64 > MAX_FOR_ITERATIONS {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!(
                    "for loop iterates {} times (max {MAX_FOR_ITERATIONS})",
                    values.len()
                ),
            });
        }

        let body_lines = split_body_into_lines(body_str);
        let mut all_instrs = Vec::new();

        for v in values {
            let substituted: Vec<String> = body_lines
                .iter()
                .map(|s| replace_word(s, &var_name, &v.to_string()))
                .collect();

            let saved = self.int_vars.clone();
            let mut new_vars = saved.clone().unwrap_or_default();
            new_vars.insert(var_name.clone(), v);
            self.int_vars = Some(new_vars);

            let lines: Vec<&str> = substituted.iter().map(String::as_str).collect();
            let was_nested = std::mem::replace(&mut self.nested, true);
            let result = self.parse_lines(&lines, line_num.saturating_sub(1));
            self.nested = was_nested;

            self.int_vars = saved;
            all_instrs.extend(result?);
        }

        Ok(all_instrs)
    }

    fn parse_barrier(&self, line: &str, line_num: usize) -> Result<Instruction> {
        let rest = line.strip_prefix("barrier").unwrap().trim();
        let mut qubits = SmallVec::<[usize; 4]>::new();
        for token in rest.split(',') {
            qubits.extend(self.resolve_qubit_arg(token.trim(), line_num)?);
        }
        Ok(Instruction::Barrier { qubits })
    }

    /// Parse `reset q[i];` or `reset q;` (broadcast over register).
    fn parse_reset(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let rest = line.strip_prefix("reset").unwrap().trim();
        if rest.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: "expected qubit argument after `reset`".to_string(),
            });
        }
        let mut out = Vec::new();
        for token in rest.split(',') {
            let qubits = self.resolve_qubit_arg(token.trim(), line_num)?;
            for q in qubits {
                out.push(Instruction::Reset { qubit: q });
            }
        }
        Ok(out)
    }

    /// Parse `if (cond) { ... }`, with an optional `else`, into guarded regions.
    ///
    /// The body reaches [`Parser::parse_lines`], so it admits any statement the
    /// parser already accepts, nested regions included. An `else` arm becomes a
    /// second region carrying the negated condition rather than a second span on
    /// the instruction; see [`Parser::lower_else_arm`] for what that costs.
    fn parse_if_block(&mut self, buf: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let (condition, body_str) = self.split_if_header(buf, line_num)?;
        let body_str = body_str.trim_start();
        let (then_src, trailing) = if body_str.starts_with('{') {
            split_braced_body(body_str, line_num, "if")?
        } else {
            split_at_else(body_str)
        };
        let then_body = self.parse_region_body(then_src, line_num)?;

        let trailing = trailing.trim();
        if trailing.is_empty() {
            return Ok(guarded(condition, then_body).into_iter().collect());
        }
        let Some(after_else) = strip_leading_keyword(trailing, "else") else {
            let word = trailing
                .split(|c: char| c.is_whitespace() || c == '(' || c == '{')
                .next()
                .unwrap_or(trailing);
            return Err(PrismError::UnsupportedConstruct {
                construct: word.to_string(),
                line: line_num,
            });
        };
        self.lower_else_arm(condition, then_body, after_else, line_num)
    }

    /// Emit the `then` region followed by a region guarded on the negated
    /// condition.
    ///
    /// The negated region re-reads the classical bits after the `then` body has
    /// run, so a `then` body that measures into its own guard bits could take
    /// both arms. That case is rejected rather than lowered.
    fn lower_else_arm(
        &mut self,
        condition: ClassicalCondition,
        then_body: Vec<Instruction>,
        after_else: &str,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        if crate::circuit::body_writes_condition_bits(&then_body, &condition) {
            return Err(PrismError::Parse {
                line: line_num,
                message: "`else` needs a condition the `if` body does not overwrite; \
                          this body measures into a bit the condition reads"
                    .to_string(),
            });
        }

        if after_else.trim().is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: "expected a statement or block after `else`".to_string(),
            });
        }
        let else_body = if let Some(nested) = strip_leading_keyword(after_else, "if") {
            self.parse_nested_else_if(nested, line_num)?
        } else if after_else.starts_with('{') {
            let (else_src, trailing) = split_braced_body(after_else, line_num, "else")?;
            if !trailing.trim().is_empty() {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!("unexpected `{}` after `else` body", trailing.trim()),
                });
            }
            self.parse_region_body(else_src, line_num)?
        } else {
            self.parse_region_body(after_else.trim_end_matches(';'), line_num)?
        };

        let mut out = Vec::new();
        out.extend(guarded(condition.clone(), then_body));
        out.extend(guarded(condition.negate(), else_body));
        Ok(out)
    }

    /// Parse the `if` of an `else if` chain, which nests one region deeper.
    fn parse_nested_else_if(&mut self, nested: &str, line_num: usize) -> Result<Vec<Instruction>> {
        self.enter_region(line_num)?;
        let parsed = if nested.contains('{') {
            self.parse_if_block(&format!("if {nested}"), line_num)
        } else {
            self.parse_if_statement(&format!("if {nested}"), line_num)
        };
        self.region_depth -= 1;
        parsed
    }

    /// Parse a region body's source text one nesting level down.
    fn parse_region_body(&mut self, src: &str, line_num: usize) -> Result<Vec<Instruction>> {
        self.enter_region(line_num)?;
        let body_lines = split_body_into_lines(src.trim());
        let refs: Vec<&str> = body_lines.iter().map(String::as_str).collect();
        let was_nested = std::mem::replace(&mut self.nested, true);
        let body = self.parse_lines(&refs, line_num.saturating_sub(1));
        self.nested = was_nested;
        self.region_depth -= 1;
        body
    }

    fn enter_region(&mut self, line_num: usize) -> Result<()> {
        if self.region_depth >= MAX_REGION_DEPTH {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("`if` blocks nest deeper than {MAX_REGION_DEPTH}"),
            });
        }
        self.region_depth += 1;
        Ok(())
    }

    /// Parse `switch (creg) { case v { .. } default { .. } }` into a chain of
    /// guarded regions, one per case label.
    ///
    /// The chain is exclusive only because no arm body may write the switched
    /// register; that is checked rather than assumed. `default` becomes the case
    /// labels' negations nested, since a conjunction has no flat form in the
    /// condition language.
    fn parse_switch_block(&mut self, buf: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let rest = buf.trim_start().strip_prefix("switch").unwrap().trim();
        let open = rest.find('(').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `(` after `switch`".to_string(),
        })?;
        let after_open = &rest[open + 1..];
        let close = find_matching_close_paren(after_open).ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `)` in `switch` operand".to_string(),
        })?;
        let (offset, size) = self.resolve_switch_operand(after_open[..close].trim(), line_num)?;
        let (arms_src, trailing) =
            split_braced_body(after_open[close + 1..].trim(), line_num, "switch")?;
        if !trailing.trim().is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("unexpected `{}` after `switch` body", trailing.trim()),
            });
        }

        let arms = split_switch_arms(arms_src, line_num)?;
        let mut labels: Vec<u64> = Vec::new();
        let mut cases: Vec<(Vec<u64>, Vec<Instruction>)> = Vec::new();
        let mut default: Option<Vec<Instruction>> = None;
        for arm in arms {
            let body = self.parse_region_body(arm.body, line_num)?;
            let Some(labels_src) = arm.labels_src else {
                if default.replace(body).is_some() {
                    return Err(PrismError::Parse {
                        line: line_num,
                        message: "`switch` has more than one `default` arm".to_string(),
                    });
                }
                continue;
            };
            let mut arm_labels = Vec::new();
            for token in labels_src.split(',') {
                let value = self.parse_switch_label(token, line_num)?;
                if labels.contains(&value) {
                    return Err(PrismError::Parse {
                        line: line_num,
                        message: format!("`switch` case label {value} appears twice"),
                    });
                }
                labels.push(value);
                arm_labels.push(value);
            }
            cases.push((arm_labels, body));
        }

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
            return Err(PrismError::Parse {
                line: line_num,
                message: "`switch` needs an operand no arm overwrites; \
                          an arm here measures into the switched register"
                    .to_string(),
            });
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
                line_num,
            )?);
        }
        Ok(out)
    }

    fn parse_switch_label(&self, token: &str, line_num: usize) -> Result<u64> {
        let value = eval_int_expr(token.trim(), line_num, self.int_vars.as_ref())?;
        u64::try_from(value).map_err(|_| PrismError::Parse {
            line: line_num,
            message: format!(
                "`switch` case label must be non-negative, got `{}`",
                token.trim()
            ),
        })
    }

    /// Resolve a `switch` operand to the `(offset, size)` of the classical bit
    /// range it names. A bit reference resolves to size 1.
    fn resolve_switch_operand(&self, operand: &str, line_num: usize) -> Result<(usize, usize)> {
        if operand.contains('[') {
            return Ok((self.resolve_cbit(operand, line_num)?, 1));
        }
        let reg = self
            .cregs
            .get(operand)
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: operand.to_string(),
                line: line_num,
            })?;
        Ok((reg.offset, reg.size))
    }

    /// Split `if (cond) rest` into the parsed condition and the trimmed rest,
    /// which is empty when the condition is the whole statement.
    fn split_if_header<'s>(
        &self,
        line: &'s str,
        line_num: usize,
    ) -> Result<(ClassicalCondition, &'s str)> {
        let rest = line.trim_start().strip_prefix("if").unwrap().trim();
        let open = rest.find('(').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `(` after `if`".to_string(),
        })?;
        let after_open = &rest[open + 1..];
        let close_offset =
            find_matching_close_paren(after_open).ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "expected `)` in `if` condition".to_string(),
            })?;
        let condition =
            self.parse_classical_condition(after_open[..close_offset].trim(), line_num)?;
        Ok((condition, after_open[close_offset + 1..].trim()))
    }

    /// Parse `if(creg==value) gate args` (OQ2) or `if (c[i]) gate args` (OQ3).
    fn parse_if_statement(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let (condition, body_str) = self.split_if_header(line, line_num)?;
        if body_str.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: "expected gate after `if(...)` condition".to_string(),
            });
        }
        let (body, input_slot) = self.parse_gate_application(body_str, line_num)?;
        if input_slot.is_some() {
            return Err(PrismError::UnsupportedConstruct {
                construct: "`input` inside a guarded statement".to_string(),
                line: line_num,
            });
        }
        Ok(guarded(condition, body).into_iter().collect())
    }

    /// Parse `a ^ b ^ c` or `(a ^ b ^ c) == 0` into a parity condition.
    ///
    /// Returns `None` when the expression holds no `^`, which leaves the
    /// register and bit forms to the caller.
    fn parse_parity_condition(
        &self,
        cond_str: &str,
        line_num: usize,
    ) -> Result<Option<ClassicalCondition>> {
        if !cond_str.contains('^') {
            return Ok(None);
        }
        let malformed = || PrismError::Parse {
            line: line_num,
            message: format!("expected `a ^ b` or `(a ^ b) == 0/1`, got: `{cond_str}`"),
        };
        let (inner, expected) = match cond_str.strip_prefix('(') {
            None => (cond_str, true),
            Some(rest) => {
                let close = find_matching_close_paren(rest).ok_or_else(malformed)?;
                let tail = rest[close + 1..].trim();
                let expected = if tail.is_empty() {
                    true
                } else {
                    let (negate, literal) = match (tail.strip_prefix("=="), tail.strip_prefix("!="))
                    {
                        (Some(literal), _) => (false, literal),
                        (_, Some(literal)) => (true, literal),
                        _ => return Err(malformed()),
                    };
                    match eval_int_expr(literal.trim(), line_num, self.int_vars.as_ref())? {
                        0 => negate,
                        1 => !negate,
                        _ => return Err(malformed()),
                    }
                };
                (&rest[..close], expected)
            }
        };

        let mut bits = Vec::new();
        for token in inner.split('^') {
            let token = token.trim();
            // `resolve_cbit` stops at the closing bracket, so a term carrying a
            // comparison would be silently truncated to its bit reference.
            if !token.ends_with(']') {
                return Err(malformed());
            }
            bits.push(self.resolve_cbit(token, line_num)?);
        }
        Ok(Some(ClassicalCondition::Parity {
            bits: bits.into(),
            expected,
        }))
    }

    /// Parse a classical condition expression for `if (...)`.
    ///
    /// Supported forms:
    /// - `c == n`, `c != n` (register vs integer)
    /// - `c[i] == 0`, `c[i] == 1`, `c[i] != 0`, `c[i] != 1` (bit vs literal)
    /// - `c[i]` (bit truthy)
    /// - `!c[i]` (bit falsy)
    /// - `c[i] ^ c[j]`, `(c[i] ^ c[j]) == 0/1` (parity over bits)
    fn parse_classical_condition(
        &self,
        cond_str: &str,
        line_num: usize,
    ) -> Result<ClassicalCondition> {
        let cond_str = cond_str.trim();

        if let Some(parity) = self.parse_parity_condition(cond_str, line_num)? {
            return Ok(parity);
        }

        if let Some(rest) = cond_str.strip_prefix('!') {
            let inner = rest.trim();
            if !inner.contains('[') {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!("expected `!c[i]` form in `if` condition, got: `{cond_str}`"),
                });
            }
            let bit = self.resolve_cbit(inner, line_num)?;
            return Ok(ClassicalCondition::BitIsZero(bit));
        }

        let (op_pos, op_len, negate) = if let Some(p) = cond_str.find("!=") {
            (Some(p), 2usize, true)
        } else if let Some(p) = cond_str.find("==") {
            (Some(p), 2usize, false)
        } else {
            (None, 0, false)
        };

        if let Some(pos) = op_pos {
            let lhs = cond_str[..pos].trim();
            let rhs = cond_str[pos + op_len..].trim();
            let value = eval_int_expr(rhs, line_num, self.int_vars.as_ref())?;
            if value < 0 {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!(
                        "negative integer in `if` condition is not supported: `{rhs}`"
                    ),
                });
            }
            let value = value as u64;

            if lhs.contains('[') {
                let bit = self.resolve_cbit(lhs, line_num)?;
                return Ok(match (value, negate) {
                    (0, false) => ClassicalCondition::BitIsZero(bit),
                    (0, true) => ClassicalCondition::BitIsOne(bit),
                    (1, false) => ClassicalCondition::BitIsOne(bit),
                    (1, true) => ClassicalCondition::BitIsZero(bit),
                    (other, _) => {
                        return Err(PrismError::Parse {
                            line: line_num,
                            message: format!(
                                "bit comparison must be against 0 or 1, got `{other}`"
                            ),
                        });
                    }
                });
            }

            let reg = self
                .cregs
                .get(lhs)
                .ok_or_else(|| PrismError::UndefinedRegister {
                    name: lhs.to_string(),
                    line: line_num,
                })?;
            return Ok(if negate {
                ClassicalCondition::RegisterNotEquals {
                    offset: reg.offset,
                    size: reg.size,
                    value,
                }
            } else {
                ClassicalCondition::RegisterEquals {
                    offset: reg.offset,
                    size: reg.size,
                    value,
                }
            });
        }

        if cond_str.contains('[') {
            let bit = self.resolve_cbit(cond_str, line_num)?;
            return Ok(ClassicalCondition::BitIsOne(bit));
        }

        Err(PrismError::Parse {
            line: line_num,
            message: format!(
                "expected `creg==value`, `creg!=value`, `c[i]`, `!c[i]`, or `c[i]==0/1` in `if` condition, got: `{cond_str}`"
            ),
        })
    }

    /// Parse one gate application, reporting the `input` slot every instruction
    /// it produced reads. A register broadcast produces several, and they share
    /// the slot, which is the weight sharing [`Parameters`] already models.
    fn parse_gate_application(
        &self,
        line: &str,
        line_num: usize,
    ) -> Result<(Vec<Instruction>, Option<usize>)> {
        let (modifiers, gate_line) = Self::strip_modifiers(line, line_num)?;

        if modifiers.is_empty() {
            if let Some(instrs) = self.try_expand_def_call(gate_line, line_num)? {
                return Ok((instrs, None));
            }
        }

        let (gate_name, params, input_slot, args_str) =
            self.split_gate_line(gate_line, line_num)?;
        if input_slot.is_some() && !modifiers.is_empty() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("modifier on `{gate_name}` reading an `input`"),
                line: line_num,
            });
        }

        let qubit_tokens: Vec<&str> = args_str
            .split(',')
            .map(|s| s.trim())
            .filter(|t| !t.is_empty())
            .collect();
        let resolved: Vec<SmallVec<[usize; 4]>> = qubit_tokens
            .iter()
            .map(|t| self.resolve_qubit_arg(t, line_num))
            .collect::<Result<Vec<_>>>()?;

        let broadcast_len = self.broadcast_length(&resolved, &gate_name, line_num)?;

        let mut all_instrs = Vec::with_capacity(broadcast_len);
        for i in 0..broadcast_len {
            let qubits: SmallVec<[usize; 4]> = resolved
                .iter()
                .map(|v| if v.len() == 1 { v[0] } else { v[i] })
                .collect();

            all_instrs.append(&mut self.resolve_gate_application_once(
                &gate_name,
                &params,
                &modifiers,
                &qubits,
                input_slot.is_some(),
                line_num,
            )?);
        }

        if input_slot.is_some() {
            Self::check_every_instruction_is_bindable(&all_instrs, &gate_name, line_num)?;
        }
        Ok((all_instrs, input_slot))
    }

    /// A slot writes one angle onto each instruction it links, so every
    /// instruction a gate reading an `input` produced has to carry one.
    fn check_every_instruction_is_bindable(
        instrs: &[Instruction],
        gate_name: &str,
        line_num: usize,
    ) -> Result<()> {
        let bindable = instrs.iter().all(|instr| {
            matches!(instr, Instruction::Gate { gate, .. } if gate.pauli_generator().is_some())
        });
        if !bindable {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`input` on `{gate_name}`, which carries no rotation angle to bind"
                ),
                line: line_num,
            });
        }
        Ok(())
    }

    fn resolve_gate_application_once(
        &self,
        gate_name: &str,
        params: &[f64],
        modifiers: &[Modifier],
        qubits: &SmallVec<[usize; 4]>,
        has_input: bool,
        line_num: usize,
    ) -> Result<Vec<Instruction>> {
        if let Some(instrs) = Self::resolve_decomposed_gate(gate_name, params, qubits, line_num)? {
            // A lowering folds the angle into its own arithmetic, so binding a
            // slot afterwards would write the raw value over a derived one.
            if has_input {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`input` on `{gate_name}`, which lowers to a gate sequence"),
                    line: line_num,
                });
            }
            if !modifiers.is_empty() {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("modifier on decomposed gate `{gate_name}`"),
                    line: line_num,
                });
            }
            return Ok(instrs);
        }

        if let Some(instrs) = self.expand_user_gate(gate_name, params, qubits, line_num)? {
            // Expansion substitutes the numeric value into the body, so a body
            // writing `rx(2 * a)` would take a bound angle whole.
            if has_input {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`input` on user-defined gate `{gate_name}`"),
                    line: line_num,
                });
            }
            return Ok(instrs);
        }

        let mut gate = Self::resolve_gate(gate_name, params, line_num)?;
        for modifier in modifiers.iter().rev() {
            gate = Self::apply_modifier(gate, modifier, line_num)?;
        }
        let expected = gate.num_qubits();
        if qubits.len() != expected {
            return Err(PrismError::GateArity {
                gate: gate_name.to_string(),
                expected,
                got: qubits.len(),
            });
        }
        Ok(vec![Instruction::Gate {
            gate,
            targets: qubits.clone(),
        }])
    }

    /// Determine the broadcast length from resolved qubit arguments.
    /// All multi-element args must have the same length. Single-element args broadcast.
    fn broadcast_length(
        &self,
        resolved: &[SmallVec<[usize; 4]>],
        gate_name: &str,
        line_num: usize,
    ) -> Result<usize> {
        let mut broadcast_len = 1usize;
        for arg in resolved {
            if arg.len() > 1 {
                if broadcast_len == 1 {
                    broadcast_len = arg.len();
                } else if arg.len() != broadcast_len {
                    return Err(PrismError::Parse {
                        line: line_num,
                        message: format!(
                            "register size mismatch in `{gate_name}`: \
                             expected {broadcast_len} qubits but got {}",
                            arg.len()
                        ),
                    });
                }
            }
        }
        Ok(broadcast_len)
    }

    /// Expand a user-defined gate by substituting parameters and qubit arguments
    /// into the gate body and recursively parsing each statement.
    fn expand_user_gate(
        &self,
        name: &str,
        call_params: &[f64],
        call_qubits: &SmallVec<[usize; 4]>,
        line_num: usize,
    ) -> Result<Option<Vec<Instruction>>> {
        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!(
                    "gate expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive gate definition for `{name}`"
                ),
            });
        }

        let def = match self.gate_defs.get(name) {
            Some(d) => d,
            None => return Ok(None),
        };

        if call_params.len() != def.params.len() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!(
                    "gate `{name}` expects {} parameters, got {}",
                    def.params.len(),
                    call_params.len()
                ),
            });
        }
        if call_qubits.len() != def.qubits.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.qubits.len(),
                got: call_qubits.len(),
            });
        }

        let mut var_map = HashMap::new();
        for (i, param_name) in def.params.iter().enumerate() {
            var_map.insert(param_name.clone(), call_params[i]);
        }

        let max_qubit = call_qubits.iter().max().copied().unwrap_or(0) + 1;
        let mut sub_parser = Parser {
            input: "",
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            gate_defs: HashMap::new(),
            def_defs: HashMap::new(),
            total_qubits: max_qubit,
            total_cbits: self.total_cbits,
            gate_expansion_depth: self.gate_expansion_depth + 1,
            region_depth: self.region_depth,
            param_vars: Some(var_map),
            int_vars: self.int_vars.clone(),
            inputs: HashMap::new(),
            input_names: Vec::new(),
            links: Vec::new(),
            pending_input_slot: None,
            nested: true,
        };
        sub_parser.qregs.insert(
            "__q__".to_string(),
            Register {
                offset: 0,
                size: max_qubit,
            },
        );
        for (k, v) in &self.qregs {
            sub_parser.qregs.insert(
                k.clone(),
                Register {
                    offset: v.offset,
                    size: v.size,
                },
            );
        }
        for (k, v) in &self.cregs {
            sub_parser.cregs.insert(
                k.clone(),
                Register {
                    offset: v.offset,
                    size: v.size,
                },
            );
        }
        for (k, v) in &self.gate_defs {
            sub_parser.gate_defs.insert(
                k.clone(),
                GateDefinition {
                    params: v.params.clone(),
                    qubits: v.qubits.clone(),
                    body: v.body.clone(),
                },
            );
        }
        self.copy_def_defs_into(&mut sub_parser);

        let mut all_instrs = Vec::new();
        for stmt in &def.body {
            let mut expanded = stmt.clone();
            for (i, qubit_name) in def.qubits.iter().enumerate() {
                expanded =
                    replace_word(&expanded, qubit_name, &format!("__q__[{}]", call_qubits[i]));
            }

            // The sub-parser declares no inputs, so the slot is always `None`.
            let (instrs, _) = sub_parser.parse_gate_application(expanded.trim(), line_num)?;
            all_instrs.extend(instrs);
        }

        Ok(Some(all_instrs))
    }

    fn copy_def_defs_into(&self, sub: &mut Parser<'_>) {
        for (k, v) in &self.def_defs {
            let cloned_args = v
                .args
                .iter()
                .map(|a| match a {
                    DefArg::Qubit(name) => DefArg::Qubit(name.clone()),
                    DefArg::Param { name, kind } => DefArg::Param {
                        name: name.clone(),
                        kind: *kind,
                    },
                })
                .collect();
            sub.def_defs.insert(
                k.clone(),
                DefDefinition {
                    args: cloned_args,
                    body: v.body.clone(),
                },
            );
        }
    }

    /// Detect and inline a `def` subroutine call of the form `name(arg1, arg2, ...)`.
    ///
    /// Returns `Ok(None)` if the line is not a known def call so the caller can
    /// fall through to standard gate-application parsing.
    fn try_expand_def_call(&self, line: &str, line_num: usize) -> Result<Option<Vec<Instruction>>> {
        let line = line.trim();
        let paren_open = match line.find('(') {
            Some(p) => p,
            None => return Ok(None),
        };
        let name = line[..paren_open].trim();
        if name.is_empty() {
            return Ok(None);
        }
        let def = match self.def_defs.get(name) {
            Some(d) => d,
            None => return Ok(None),
        };

        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!(
                    "def expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive call to `{name}`"
                ),
            });
        }

        let after_open = &line[paren_open + 1..];
        let close = find_matching_close_paren(after_open).ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: format!("unmatched `(` in def call `{name}`"),
        })?;
        let args_str = &after_open[..close];
        let trailing = after_open[close + 1..].trim();
        let trailing = trailing.strip_suffix(';').unwrap_or(trailing).trim();
        if !trailing.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!("unexpected tokens after def call `{name}(...)`: `{trailing}`"),
            });
        }

        let raw_args: Vec<&str> = split_top_level_commas(args_str)
            .into_iter()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if raw_args.len() != def.args.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.args.len(),
                got: raw_args.len(),
            });
        }

        // The body is inlined with the argument's numeric value substituted in,
        // so a slot recorded here would bind an angle the body derives from.
        // Rejecting by name beats the unknown-identifier error the substitution
        // would otherwise raise on a name that is declared.
        if let Some(input) = self
            .input_names
            .iter()
            .find(|input| raw_args.iter().any(|arg| contains_word(arg, input)))
        {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("input `{input}` as an argument to `def {name}`"),
                line: line_num,
            });
        }

        let mut qubit_substs: Vec<(String, usize)> = Vec::new();
        let mut float_vars: HashMap<String, f64> = self.param_vars.clone().unwrap_or_default();
        let mut int_vars: HashMap<String, i64> = self.int_vars.clone().unwrap_or_default();
        let mut int_substs: Vec<(String, i64)> = Vec::new();

        for (slot, arg) in def.args.iter().zip(raw_args.iter()) {
            match slot {
                DefArg::Qubit(param_name) => {
                    let resolved = self.resolve_qubit_arg(arg, line_num)?;
                    if resolved.len() != 1 {
                        return Err(PrismError::Parse {
                            line: line_num,
                            message: format!(
                                "def `{name}` qubit parameter `{param_name}` requires a single qubit, got register `{arg}`"
                            ),
                        });
                    }
                    qubit_substs.push((param_name.clone(), resolved[0]));
                }
                DefArg::Param { name: pname, kind } => match kind {
                    DefParamKind::Float => {
                        let val = eval_expr(arg, line_num, Some(&float_vars))?;
                        float_vars.insert(pname.clone(), val);
                    }
                    DefParamKind::Int => {
                        let val = eval_int_expr(arg, line_num, Some(&int_vars))?;
                        int_vars.insert(pname.clone(), val);
                        float_vars.insert(pname.clone(), val as f64);
                        int_substs.push((pname.clone(), val));
                    }
                },
            }
        }

        let max_qubit = qubit_substs
            .iter()
            .map(|(_, q)| *q)
            .max()
            .unwrap_or(0)
            .max(self.total_qubits.saturating_sub(1))
            + 1;

        let mut sub_parser = Parser {
            input: "",
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            gate_defs: HashMap::new(),
            def_defs: HashMap::new(),
            total_qubits: max_qubit,
            total_cbits: self.total_cbits,
            gate_expansion_depth: self.gate_expansion_depth + 1,
            region_depth: self.region_depth,
            param_vars: Some(float_vars),
            int_vars: Some(int_vars),
            inputs: HashMap::new(),
            input_names: Vec::new(),
            links: Vec::new(),
            pending_input_slot: None,
            nested: true,
        };
        sub_parser.qregs.insert(
            "__q__".to_string(),
            Register {
                offset: 0,
                size: max_qubit,
            },
        );
        for (k, v) in &self.qregs {
            sub_parser.qregs.insert(
                k.clone(),
                Register {
                    offset: v.offset,
                    size: v.size,
                },
            );
        }
        for (k, v) in &self.cregs {
            sub_parser.cregs.insert(
                k.clone(),
                Register {
                    offset: v.offset,
                    size: v.size,
                },
            );
        }
        for (k, v) in &self.gate_defs {
            sub_parser.gate_defs.insert(
                k.clone(),
                GateDefinition {
                    params: v.params.clone(),
                    qubits: v.qubits.clone(),
                    body: v.body.clone(),
                },
            );
        }
        self.copy_def_defs_into(&mut sub_parser);

        let mut substituted: Vec<String> = Vec::with_capacity(def.body.len());
        for stmt in &def.body {
            let mut expanded = stmt.clone();
            for (qname, qidx) in &qubit_substs {
                expanded = replace_word(&expanded, qname, &format!("__q__[{}]", qidx));
            }
            for (iname, ival) in &int_substs {
                expanded = replace_word(&expanded, iname, &ival.to_string());
            }
            substituted.push(expanded);
        }

        let lines: Vec<&str> = substituted.iter().map(String::as_str).collect();
        let instrs = sub_parser.parse_lines(&lines, line_num.saturating_sub(1))?;
        Ok(Some(instrs))
    }

    fn strip_modifiers(line: &str, line_num: usize) -> Result<(Vec<Modifier>, &str)> {
        if !line.contains(" @ ") {
            return Ok((vec![], line));
        }
        let parts: Vec<&str> = line.split(" @ ").collect();
        let gate_line = parts[parts.len() - 1];
        let mut modifiers = Vec::with_capacity(parts.len() - 1);
        for part in &parts[..parts.len() - 1] {
            let token = part.trim();
            if token == "inv" {
                modifiers.push(Modifier::Inv);
            } else if token == "ctrl" {
                modifiers.push(Modifier::Ctrl);
            } else if let Some(rest) = token.strip_prefix("pow(") {
                let rest = rest.strip_suffix(')').ok_or_else(|| PrismError::Parse {
                    line: line_num,
                    message: format!("unmatched `(` in pow modifier: `{token}`"),
                })?;
                let k: i64 = rest
                    .trim()
                    .parse()
                    .map_err(|_| PrismError::UnsupportedConstruct {
                        construct: format!("pow({rest})"),
                        line: line_num,
                    })?;
                modifiers.push(Modifier::Pow(k));
            } else {
                return Err(PrismError::UnsupportedConstruct {
                    construct: token.to_string(),
                    line: line_num,
                });
            }
        }
        Ok((modifiers, gate_line))
    }

    fn apply_modifier(gate: Gate, modifier: &Modifier, line_num: usize) -> Result<Gate> {
        match modifier {
            Modifier::Inv => Ok(gate.inverse()),
            Modifier::Pow(k) => {
                if gate.num_qubits() != 1 {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("pow({k}) @ {} (only single-qubit gates)", gate.name()),
                        line: line_num,
                    });
                }
                Ok(gate.matrix_power(*k))
            }
            Modifier::Ctrl => match &gate {
                g if g.num_qubits() == 1 => {
                    let mat = gate.matrix_2x2();
                    Ok(Self::resolve_controlled(mat))
                }
                Gate::Cu(mat) => Ok(Gate::mcu(**mat, 2)),
                Gate::Cx => Ok(Gate::mcu(Gate::X.matrix_2x2(), 2)),
                Gate::Cz => Ok(Gate::mcu(Gate::Z.matrix_2x2(), 2)),
                Gate::Mcu(data) => {
                    let num_controls = data.num_controls.checked_add(1).ok_or_else(|| {
                        PrismError::UnsupportedConstruct {
                            construct: format!("ctrl @ chain past {} controls", u8::MAX),
                            line: line_num,
                        }
                    })?;
                    Ok(Gate::mcu(data.mat, num_controls))
                }
                _ => Err(PrismError::UnsupportedConstruct {
                    construct: format!("ctrl @ {} (unsupported gate type)", gate.name()),
                    line: line_num,
                }),
            },
        }
    }

    fn resolve_controlled(mat: [[num_complex::Complex64; 2]; 2]) -> Gate {
        use num_complex::Complex64;
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let eps = 1e-12;

        // CX: mat = X = [[0,1],[1,0]]
        if (mat[0][0] - zero).norm() < eps
            && (mat[0][1] - one).norm() < eps
            && (mat[1][0] - one).norm() < eps
            && (mat[1][1] - zero).norm() < eps
        {
            return Gate::Cx;
        }
        // CZ: mat = Z = [[1,0],[0,-1]]
        if (mat[0][0] - one).norm() < eps
            && (mat[0][1] - zero).norm() < eps
            && (mat[1][0] - zero).norm() < eps
            && (mat[1][1] + one).norm() < eps
        {
            return Gate::Cz;
        }
        Gate::cu(mat)
    }

    fn split_gate_line(
        &self,
        line: &str,
        line_num: usize,
    ) -> Result<(String, Vec<f64>, Option<usize>, String)> {
        if let Some(paren_start) = line.find('(') {
            let mut depth = 0usize;
            let mut paren_end = None;
            for (i, ch) in line[paren_start..].char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            paren_end = Some(paren_start + i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let paren_end = paren_end.ok_or_else(|| PrismError::Parse {
                line: line_num,
                message: "unmatched `(` in gate application".to_string(),
            })?;
            let gate_name = line[..paren_start].trim().to_string();
            let params_str = &line[paren_start + 1..paren_end];
            let (params, input_slot) = self.parse_params(params_str, line_num)?;
            let args_str = line[paren_end + 1..].trim().to_string();
            Ok((gate_name, params, input_slot, args_str))
        } else {
            let first_space = line
                .find(char::is_whitespace)
                .ok_or_else(|| PrismError::Parse {
                    line: line_num,
                    message: format!("cannot parse instruction: `{line}`"),
                })?;
            let gate_name = line[..first_space].trim().to_string();
            let args_str = line[first_space..].trim().to_string();
            Ok((gate_name, vec![], None, args_str))
        }
    }

    /// Evaluate a gate's argument list, reporting the one `input` slot it reads.
    ///
    /// Two inputs on one gate are rejected: a slot is written onto the single
    /// angle a bindable gate carries, so a second one would have nowhere to go.
    fn parse_params(&self, params_str: &str, line_num: usize) -> Result<(Vec<f64>, Option<usize>)> {
        let mut values = Vec::new();
        let mut input_slot = None;
        for part in split_top_level_commas(params_str) {
            let (value, slot) = self.resolve_param(part, line_num)?;
            if slot.is_some() && input_slot.is_some() {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "two `input` parameters on one gate".to_string(),
                    line: line_num,
                });
            }
            input_slot = input_slot.or(slot);
            values.push(value);
        }
        Ok((values, input_slot))
    }

    fn resolve_gate(name: &str, params: &[f64], line_num: usize) -> Result<Gate> {
        match name {
            "id" => Ok(Gate::Id),
            "x" => Ok(Gate::X),
            "y" => Ok(Gate::Y),
            "z" => Ok(Gate::Z),
            "h" => Ok(Gate::H),
            "s" => Ok(Gate::S),
            "sdg" => Ok(Gate::Sdg),
            "t" => Ok(Gate::T),
            "tdg" => Ok(Gate::Tdg),
            "rx" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Rx(params[0]))
            }
            "ry" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Ry(params[0]))
            }
            "rz" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Rz(params[0]))
            }
            "p" | "phase" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::P(params[0]))
            }
            "r" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Ok(Gate::Fused(Box::new(Self::r_matrix(params[0], params[1]))))
            }
            "sx" => Ok(Gate::SX),
            "sxdg" => Ok(Gate::SXdg),
            "cp" | "cphase" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cphase(params[0]))
            }
            "rzz" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Rzz(params[0]))
            }
            "cx" | "CX" | "cnot" => Ok(Gate::Cx),
            "cy" => Ok(Gate::cu(Gate::Y.matrix_2x2())),
            "cs" => Ok(Gate::cu(Gate::S.matrix_2x2())),
            "csdg" => Ok(Gate::cu(Gate::Sdg.matrix_2x2())),
            "ch" => Ok(Gate::cu(Gate::H.matrix_2x2())),
            "cu" => {
                Self::expect_param_count(name, params, 4, line_num)?;
                Ok(Gate::cu(Self::cu_target_matrix(
                    params[0], params[1], params[2], params[3],
                )))
            }
            "crx" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cu(Gate::Rx(params[0]).matrix_2x2()))
            }
            "cry" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cu(Gate::Ry(params[0]).matrix_2x2()))
            }
            "crz" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::cu(Gate::Rz(params[0]).matrix_2x2()))
            }
            "csx" => Ok(Gate::cu(Gate::SX.matrix_2x2())),
            "cz" => Ok(Gate::Cz),
            "swap" => Ok(Gate::Swap),
            "ccx" | "toffoli" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 2)),
            "ccz" => Ok(Gate::mcu(Gate::Z.matrix_2x2(), 2)),
            "c3x" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 3)),
            "c4x" => Ok(Gate::mcu(Gate::X.matrix_2x2(), 4)),
            "xx_plus_yy" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::xx_plus_yy_matrix(
                    params[0], params[1],
                ))))
            }
            "xx_minus_yy" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Ok(Gate::Fused2q(Box::new(Self::xx_minus_yy_matrix(
                    params[0], params[1],
                ))))
            }
            "gpi" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Fused(Box::new(Self::gpi_matrix(params[0]))))
            }
            "gpi2" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::Fused(Box::new(Self::gpi2_matrix(params[0]))))
            }
            "ms" => {
                if !(params.len() == 2 || params.len() == 3) {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "`{name}` at line {line_num} requires 2 or 3 parameter(s), got {}",
                            params.len()
                        ),
                    });
                }
                let theta = params.get(2).copied().unwrap_or(0.25);
                Ok(Gate::Fused2q(Box::new(Self::ms_matrix(
                    params[0], params[1], theta,
                ))))
            }
            "syc" => Ok(Gate::Fused2q(Box::new(Self::syc_matrix()))),
            "sqrt_iswap" => Ok(Gate::Fused2q(Box::new(Self::sqrt_iswap_matrix(1.0)))),
            "sqrt_iswap_inv" => Ok(Gate::Fused2q(Box::new(Self::sqrt_iswap_matrix(-1.0)))),
            _ => Err(PrismError::UnsupportedConstruct {
                construct: name.to_string(),
                line: line_num,
            }),
        }
    }

    /// Decomposition-body shorthand for a gate instruction.
    fn ig(gate: Gate, targets: &[usize]) -> Instruction {
        Instruction::Gate {
            gate,
            targets: SmallVec::from_slice(targets),
        }
    }

    /// Handle gates that decompose into multiple instructions at parse time.
    ///
    /// Returns `Ok(None)` if the gate name is not a decomposed gate (caller
    /// should fall through to `resolve_gate`).
    fn resolve_decomposed_gate(
        name: &str,
        params: &[f64],
        qubits: &[usize],
        line_num: usize,
    ) -> Result<Option<Vec<Instruction>>> {
        match name {
            "mcx" => {
                if qubits.len() < 2 {
                    return Err(PrismError::GateArity {
                        gate: name.to_string(),
                        expected: 2,
                        got: qubits.len(),
                    });
                }
                let controls = qubits.len() - 1;
                if controls > u8::MAX as usize {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "`{name}` at line {line_num} supports at most {} controls, got {controls}",
                            u8::MAX
                        ),
                    });
                }
                Ok(Some(vec![Self::ig(
                    Gate::mcu(Gate::X.matrix_2x2(), controls as u8),
                    qubits,
                )]))
            }
            "rccx" => {
                Self::check_arity(name, qubits, 3)?;
                let c0 = qubits[0];
                let c1 = qubits[1];
                let target = qubits[2];
                Ok(Some(vec![
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::Cx, &[c0, target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                ]))
            }
            "rc3x" | "rcccx" => {
                Self::check_arity(name, qubits, 4)?;
                let c0 = qubits[0];
                let c1 = qubits[1];
                let c2 = qubits[2];
                let target = qubits[3];
                Ok(Some(vec![
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c2, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::Cx, &[c0, target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::Cx, &[c0, target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c1, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                    Self::ig(Gate::T, &[target]),
                    Self::ig(Gate::Cx, &[c2, target]),
                    Self::ig(Gate::Tdg, &[target]),
                    Self::ig(Gate::H, &[target]),
                ]))
            }
            "cswap" | "fredkin" => {
                Self::check_arity(name, qubits, 3)?;
                let ctrl = qubits[0];
                let t1 = qubits[1];
                let t2 = qubits[2];
                Ok(Some(vec![
                    Self::ig(Gate::Cx, &[t2, t1]),
                    Self::ig(Gate::mcu(Gate::X.matrix_2x2(), 2), &[ctrl, t1, t2]),
                    Self::ig(Gate::Cx, &[t2, t1]),
                ]))
            }
            "rxx" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Self::ig(Gate::H, &[q0]),
                    Self::ig(Gate::H, &[q1]),
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::Rz(params[0]), &[q1]),
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::H, &[q0]),
                    Self::ig(Gate::H, &[q1]),
                ]))
            }
            "ryy" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                let half_pi = std::f64::consts::FRAC_PI_2;
                Ok(Some(vec![
                    Self::ig(Gate::Rx(half_pi), &[q0]),
                    Self::ig(Gate::Rx(half_pi), &[q1]),
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::Rz(params[0]), &[q1]),
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::Rx(-half_pi), &[q0]),
                    Self::ig(Gate::Rx(-half_pi), &[q1]),
                ]))
            }
            "ecr" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Self::ig(Gate::Rz(std::f64::consts::FRAC_PI_4), &[q0]),
                    Self::ig(Gate::Rx(std::f64::consts::FRAC_PI_2), &[q0]),
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::X, &[q0]),
                ]))
            }
            "iswap" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Self::ig(Gate::S, &[q0]),
                    Self::ig(Gate::S, &[q1]),
                    Self::ig(Gate::H, &[q0]),
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::Cx, &[q1, q0]),
                    Self::ig(Gate::H, &[q1]),
                ]))
            }
            "dcx" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Self::ig(Gate::Cx, &[q0, q1]),
                    Self::ig(Gate::Cx, &[q1, q0]),
                ]))
            }
            "u1" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                Ok(Some(vec![Self::ig(Gate::P(params[0]), &[qubits[0]])]))
            }
            "u2" => {
                Self::expect_param_count(name, params, 2, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                let phi = params[0];
                let lam = params[1];
                let isqrt2 = std::f64::consts::FRAC_1_SQRT_2;
                let one = Complex64::new(isqrt2, 0.0);
                let mat = [
                    [one, -Complex64::from_polar(isqrt2, lam)],
                    [
                        Complex64::from_polar(isqrt2, phi),
                        Complex64::from_polar(isqrt2, phi + lam),
                    ],
                ];
                Ok(Some(vec![Self::ig(
                    Gate::Fused(Box::new(mat)),
                    &[qubits[0]],
                )]))
            }
            "u3" | "u" | "U" => {
                Self::expect_param_count(name, params, 3, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                let theta = params[0];
                let phi = params[1];
                let lam = params[2];
                let mat = Self::u_matrix(theta, phi, lam);
                Ok(Some(vec![Self::ig(
                    Gate::Fused(Box::new(mat)),
                    &[qubits[0]],
                )]))
            }
            _ => Ok(None),
        }
    }

    fn check_arity(name: &str, qubits: &[usize], expected: usize) -> Result<()> {
        if qubits.len() != expected {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected,
                got: qubits.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn u_matrix(theta: f64, phi: f64, lam: f64) -> [[Complex64; 2]; 2] {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        [
            [Complex64::new(c, 0.0), -Complex64::from_polar(s, lam)],
            [
                Complex64::from_polar(s, phi),
                Complex64::from_polar(c, phi + lam),
            ],
        ]
    }

    fn r_matrix(theta: f64, phi: f64) -> [[Complex64; 2]; 2] {
        let zero_phase = Complex64::new((theta / 2.0).cos(), 0.0);
        let off = Complex64::new(0.0, -1.0) * (theta / 2.0).sin();
        [
            [zero_phase, off * Complex64::from_polar(1.0, -phi)],
            [off * Complex64::from_polar(1.0, phi), zero_phase],
        ]
    }

    pub(crate) fn cu_target_matrix(
        theta: f64,
        phi: f64,
        lam: f64,
        gamma: f64,
    ) -> [[Complex64; 2]; 2] {
        let phase = Complex64::from_polar(1.0, gamma);
        let u = Self::u_matrix(theta, phi, lam);
        [
            [phase * u[0][0], phase * u[0][1]],
            [phase * u[1][0], phase * u[1][1]],
        ]
    }

    pub(crate) fn xx_plus_yy_matrix(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let c = Complex64::new((theta / 2.0).cos(), 0.0);
        let s = Complex64::new(0.0, -(theta / 2.0).sin());
        [
            [one, zero, zero, zero],
            [zero, c, s * Complex64::from_polar(1.0, -beta), zero],
            [zero, s * Complex64::from_polar(1.0, beta), c, zero],
            [zero, zero, zero, one],
        ]
    }

    pub(crate) fn xx_minus_yy_matrix(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let c = Complex64::new((theta / 2.0).cos(), 0.0);
        let s = Complex64::new(0.0, -(theta / 2.0).sin());
        [
            [c, zero, zero, s * Complex64::from_polar(1.0, -beta)],
            [zero, one, zero, zero],
            [zero, zero, one, zero],
            [s * Complex64::from_polar(1.0, beta), zero, zero, c],
        ]
    }

    fn gpi_matrix(phi: f64) -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        [
            [
                zero,
                Complex64::from_polar(1.0, -std::f64::consts::TAU * phi),
            ],
            [
                Complex64::from_polar(1.0, std::f64::consts::TAU * phi),
                zero,
            ],
        ]
    }

    fn gpi2_matrix(phi: f64) -> [[Complex64; 2]; 2] {
        let one = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let off = Complex64::new(0.0, -std::f64::consts::FRAC_1_SQRT_2);
        [
            [
                one,
                off * Complex64::from_polar(1.0, -std::f64::consts::TAU * phi),
            ],
            [
                off * Complex64::from_polar(1.0, std::f64::consts::TAU * phi),
                one,
            ],
        ]
    }

    pub(crate) fn ms_matrix(phi0: f64, phi1: f64, theta: f64) -> [[Complex64; 4]; 4] {
        let zero = Complex64::new(0.0, 0.0);
        let c = Complex64::new((std::f64::consts::PI * theta).cos(), 0.0);
        let s = Complex64::new(0.0, -(std::f64::consts::PI * theta).sin());
        let sum = phi0 + phi1;
        let diff = phi0 - phi1;
        [
            [
                c,
                zero,
                zero,
                s * Complex64::from_polar(1.0, -std::f64::consts::TAU * sum),
            ],
            [
                zero,
                c,
                s * Complex64::from_polar(1.0, -std::f64::consts::TAU * diff),
                zero,
            ],
            [
                zero,
                s * Complex64::from_polar(1.0, std::f64::consts::TAU * diff),
                c,
                zero,
            ],
            [
                s * Complex64::from_polar(1.0, std::f64::consts::TAU * sum),
                zero,
                zero,
                c,
            ],
        ]
    }

    pub(crate) fn syc_matrix() -> [[Complex64; 4]; 4] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let neg_i = Complex64::new(0.0, -1.0);
        [
            [one, zero, zero, zero],
            [zero, zero, neg_i, zero],
            [zero, neg_i, zero, zero],
            [
                zero,
                zero,
                zero,
                Complex64::from_polar(1.0, -std::f64::consts::PI / 6.0),
            ],
        ]
    }

    pub(crate) fn sqrt_iswap_matrix(sign: f64) -> [[Complex64; 4]; 4] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let half = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let off = Complex64::new(0.0, sign * std::f64::consts::FRAC_1_SQRT_2);
        [
            [one, zero, zero, zero],
            [zero, half, off, zero],
            [zero, off, half, zero],
            [zero, zero, zero, one],
        ]
    }

    fn expect_param_count(
        gate: &str,
        params: &[f64],
        expected: usize,
        line_num: usize,
    ) -> Result<()> {
        if params.len() != expected {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "`{gate}` at line {line_num} requires {expected} parameter(s), got {}",
                    params.len()
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "openqasm_tests.rs"]
mod tests;
