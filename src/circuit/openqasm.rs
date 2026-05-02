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
//! - `defcal`, `extern`, `opaque`, `box`, `while`
//! - `def` bodies that contain `measure`, `reset`, `bit`, `creg`, `return`,
//!   or the `=measure` assignment shape (V1 supports unitary subroutines only)
//! - `def` declarations with a return type
//! - `ctrl @ swap` modifier form (use `cswap` or `fredkin` keyword instead)
//! - `pow(k) @` with non-integer k (fractional powers)
//! - Bit literal comparisons against integers other than `0` / `1`
//! - Negative integer literals in `if` register comparisons
//! - `duration`, `stretch` outside `def` parameter lists
//!
//! # Error behaviour
//!
//! All parse failures return `PrismError::Parse` or `PrismError::UnsupportedConstruct`
//! with the source line number. The parser never panics on user input.

use num_complex::Complex64;

use crate::circuit::{smallvec, Circuit, ClassicalCondition, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use std::collections::HashMap;

/// Parse an OpenQASM 3.0 string into a PRISM-Q [`Circuit`].
///
/// This is the primary input entrypoint. The entire parse happens in-memory
/// from the provided `&str` — no file I/O.
///
/// # Errors
///
/// Returns structured [`PrismError`] for any parse failure or unsupported construct.
pub fn parse(input: &str) -> Result<Circuit> {
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
}

struct BlockState {
    kind: BlockKind,
    buf: String,
    start_line: usize,
    depth: usize,
}

struct Parser<'a> {
    input: &'a str,
    qregs: HashMap<String, Register>,
    cregs: HashMap<String, Register>,
    gate_defs: HashMap<String, GateDefinition>,
    def_defs: HashMap<String, DefDefinition>,
    total_qubits: usize,
    total_cbits: usize,
    gate_expansion_depth: usize,
    param_vars: Option<HashMap<String, f64>>,
    int_vars: Option<HashMap<String, i64>>,
}

const MAX_GATE_EXPANSION_DEPTH: usize = 32;
const MAX_FOR_ITERATIONS: i64 = 1_000_000;

use super::expr::{eval_expr, replace_word, split_top_level_commas};

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
            param_vars: None,
            int_vars: None,
        }
    }

    fn parse(mut self) -> Result<Circuit> {
        let lines: Vec<&str> = self.input.lines().collect();
        let instructions = self.parse_lines(&lines, 0)?;

        Ok(Circuit {
            num_qubits: self.total_qubits,
            num_classical_bits: self.total_cbits,
            instructions,
        })
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
                if state.depth == 0 {
                    let finished = block.take().unwrap();
                    instructions.extend(self.dispatch_block(&finished)?);
                }
                continue;
            }

            instructions.extend(self.process_top_line(line, line_num, &mut block)?);
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
        let first_word = line
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or(line);

        if matches!(first_word, "gate" | "def" | "for") {
            let kind = match first_word {
                "gate" => BlockKind::Gate,
                "def" => BlockKind::Def,
                "for" => BlockKind::For,
                _ => unreachable!(),
            };
            let depth = update_brace_depth(0, line);
            if line.contains('{') && depth == 0 {
                let state = BlockState {
                    kind,
                    buf: line.to_string(),
                    start_line: line_num,
                    depth: 0,
                };
                return self.dispatch_block(&state);
            }
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
                depth,
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
            return self.parse_if_statement(line, line_num);
        }

        if matches!(
            first_word,
            "defcal" | "opaque" | "while" | "box" | "extern" | "return"
        ) {
            return Err(PrismError::UnsupportedConstruct {
                construct: first_word.to_string(),
                line: line_num,
            });
        }

        self.parse_gate_application(line, line_num)
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
        }
    }

    /// OQ3 syntax: `qubit[4] q` or `qubit q` (single qubit).
    fn parse_qubit_decl(&mut self, line: &str, line_num: usize) -> Result<()> {
        let rest = line.strip_prefix("qubit").unwrap();

        if rest.trim_start().starts_with('[') {
            let bracket_content = Self::extract_bracket(rest).ok_or(PrismError::Parse {
                line: line_num,
                message: "missing `]` in qubit declaration".to_string(),
            })?;
            let size: usize = bracket_content.parse().map_err(|_| PrismError::Parse {
                line: line_num,
                message: format!("invalid qubit count: `{bracket_content}`"),
            })?;
            if size == 0 {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: "qubit count must be > 0".to_string(),
                });
            }
            let end = rest.find(']').unwrap(); // safe: extract_bracket succeeded
            let after_bracket = rest[end + 1..].trim();
            let name = after_bracket.to_string();
            if name.is_empty() {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: "qubit declaration missing name".to_string(),
                });
            }
            let offset = self.total_qubits;
            self.total_qubits += size;
            self.qregs.insert(name, Register { offset, size });
        } else {
            let name = rest.trim().to_string();
            if name.is_empty() {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: "qubit declaration missing name".to_string(),
                });
            }
            let offset = self.total_qubits;
            self.total_qubits += 1;
            self.qregs.insert(name, Register { offset, size: 1 });
        }
        Ok(())
    }

    /// OQ3 syntax: `bit[4] c` or `bit c` (single bit).
    fn parse_bit_decl(&mut self, line: &str, line_num: usize) -> Result<()> {
        let rest = line.strip_prefix("bit").unwrap();

        if rest.trim_start().starts_with('[') {
            let bracket_content = Self::extract_bracket(rest).ok_or(PrismError::Parse {
                line: line_num,
                message: "missing `]` in bit declaration".to_string(),
            })?;
            let size: usize = bracket_content.parse().map_err(|_| PrismError::Parse {
                line: line_num,
                message: format!("invalid bit count: `{bracket_content}`"),
            })?;
            if size == 0 {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: "bit count must be > 0".to_string(),
                });
            }
            let end = rest.find(']').unwrap(); // safe: extract_bracket succeeded
            let after_bracket = rest[end + 1..].trim();
            let name = after_bracket.to_string();
            if name.is_empty() {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: "bit declaration missing name".to_string(),
                });
            }
            let offset = self.total_cbits;
            self.total_cbits += size;
            self.cregs.insert(name, Register { offset, size });
        } else {
            let name = rest.trim().to_string();
            if name.is_empty() {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: "bit declaration missing name".to_string(),
                });
            }
            let offset = self.total_cbits;
            self.total_cbits += 1;
            self.cregs.insert(name, Register { offset, size: 1 });
        }
        Ok(())
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

    fn resolve_qubit(&self, token: &str, line_num: usize) -> Result<usize> {
        let (name, idx) = self.parse_indexed_ref(token, line_num)?;
        let reg = self
            .qregs
            .get(name)
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: name.to_string(),
                line: line_num,
            })?;
        if idx >= reg.size {
            return Err(PrismError::InvalidQubit {
                index: idx,
                register_size: reg.size,
            });
        }
        Ok(reg.offset + idx)
    }

    /// Resolve a qubit argument that may be indexed (`q[0]`) or a bare register (`q`).
    /// Returns all matching qubit indices.
    fn resolve_qubit_arg(&self, token: &str, line_num: usize) -> Result<SmallVec<[usize; 4]>> {
        if token.contains('[') {
            Ok(smallvec![self.resolve_qubit(token, line_num)?])
        } else {
            let reg = self
                .qregs
                .get(token)
                .ok_or_else(|| PrismError::UndefinedRegister {
                    name: token.to_string(),
                    line: line_num,
                })?;
            Ok((0..reg.size).map(|i| reg.offset + i).collect())
        }
    }

    /// Resolve a classical bit argument that may be indexed (`c[0]`) or a bare register (`c`).
    fn resolve_cbit_arg(&self, token: &str, line_num: usize) -> Result<SmallVec<[usize; 4]>> {
        if token.contains('[') {
            Ok(smallvec![self.resolve_cbit(token, line_num)?])
        } else {
            let reg = self
                .cregs
                .get(token)
                .ok_or_else(|| PrismError::UndefinedRegister {
                    name: token.to_string(),
                    line: line_num,
                })?;
            Ok((0..reg.size).map(|i| reg.offset + i).collect())
        }
    }

    fn resolve_cbit(&self, token: &str, line_num: usize) -> Result<usize> {
        let (name, idx) = self.parse_indexed_ref(token, line_num)?;
        let reg = self
            .cregs
            .get(name)
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: name.to_string(),
                line: line_num,
            })?;
        if idx >= reg.size {
            return Err(PrismError::InvalidClassicalBit {
                index: idx,
                register_size: reg.size,
            });
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
            .iter()
            .zip(cbits.iter())
            .map(|(&qubit, &classical_bit)| Instruction::Measure {
                qubit,
                classical_bit,
            })
            .collect())
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
            .iter()
            .zip(cbits.iter())
            .map(|(&qubit, &classical_bit)| Instruction::Measure {
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
            let name = parts[0].to_string();
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
            let result = self.parse_lines(&lines, line_num.saturating_sub(1));

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

    /// Parse `if(creg==value) gate args` (OQ2) or `if (c[i]) gate args` (OQ3).
    fn parse_if_statement(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let rest = line.strip_prefix("if").unwrap().trim();
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
        let cond_str = after_open[..close_offset].trim();
        let body_str = after_open[close_offset + 1..].trim();

        if body_str.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: "expected gate after `if(...)` condition".to_string(),
            });
        }

        let condition = self.parse_classical_condition(cond_str, line_num)?;

        let gate_instrs = self.parse_gate_application(body_str, line_num)?;
        Ok(gate_instrs
            .into_iter()
            .map(|inst| match inst {
                Instruction::Gate { gate, targets } => Instruction::Conditional {
                    condition: condition.clone(),
                    gate,
                    targets,
                },
                other => other,
            })
            .collect())
    }

    /// Parse a classical condition expression for `if (...)`.
    ///
    /// Supported forms:
    /// - `c == n`, `c != n` (register vs integer)
    /// - `c[i] == 0`, `c[i] == 1`, `c[i] != 0`, `c[i] != 1` (bit vs literal)
    /// - `c[i]` (bit truthy)
    /// - `!c[i]` (bit falsy)
    fn parse_classical_condition(
        &self,
        cond_str: &str,
        line_num: usize,
    ) -> Result<ClassicalCondition> {
        let cond_str = cond_str.trim();

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

    fn parse_gate_application(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let (modifiers, gate_line) = Self::strip_modifiers(line, line_num)?;

        if modifiers.is_empty() {
            if let Some(instrs) = self.try_expand_def_call(gate_line, line_num)? {
                return Ok(instrs);
            }
        }

        let (gate_name, params, args_str) = self.split_gate_line(gate_line, line_num)?;

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

        if broadcast_len <= 1 {
            let qubits: SmallVec<[usize; 4]> = resolved.iter().map(|v| v[0]).collect();

            if let Some(instrs) =
                Self::resolve_decomposed_gate(&gate_name, &params, &qubits, line_num)?
            {
                if !modifiers.is_empty() {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("modifier on decomposed gate `{gate_name}`"),
                        line: line_num,
                    });
                }
                return Ok(instrs);
            }

            if let Some(instrs) = self.expand_user_gate(&gate_name, &params, &qubits, line_num)? {
                return Ok(instrs);
            }

            let mut gate = Self::resolve_gate(&gate_name, &params, line_num)?;
            for modifier in modifiers.iter().rev() {
                gate = Self::apply_modifier(gate, modifier, line_num)?;
            }
            let expected = gate.num_qubits();
            if qubits.len() != expected {
                return Err(PrismError::GateArity {
                    gate: gate_name,
                    expected,
                    got: qubits.len(),
                });
            }
            return Ok(vec![Instruction::Gate {
                gate,
                targets: qubits,
            }]);
        }

        let mut all_instrs = Vec::with_capacity(broadcast_len);
        for i in 0..broadcast_len {
            let qubits: SmallVec<[usize; 4]> = resolved
                .iter()
                .map(|v| if v.len() == 1 { v[0] } else { v[i] })
                .collect();

            if let Some(mut instrs) =
                Self::resolve_decomposed_gate(&gate_name, &params, &qubits, line_num)?
            {
                if !modifiers.is_empty() {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("modifier on decomposed gate `{gate_name}`"),
                        line: line_num,
                    });
                }
                all_instrs.append(&mut instrs);
                continue;
            }

            if let Some(mut instrs) =
                self.expand_user_gate(&gate_name, &params, &qubits, line_num)?
            {
                all_instrs.append(&mut instrs);
                continue;
            }

            let mut gate = Self::resolve_gate(&gate_name, &params, line_num)?;
            for modifier in modifiers.iter().rev() {
                gate = Self::apply_modifier(gate, modifier, line_num)?;
            }
            all_instrs.push(Instruction::Gate {
                gate,
                targets: qubits,
            });
        }
        Ok(all_instrs)
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

        let mut all_instrs = Vec::new();
        for stmt in &def.body {
            let mut expanded = stmt.clone();
            for (i, qubit_name) in def.qubits.iter().enumerate() {
                expanded =
                    replace_word(&expanded, qubit_name, &format!("__q__[{}]", call_qubits[i]));
            }

            let saved_qregs = &self.qregs;
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
                param_vars: Some(var_map.clone()),
                int_vars: self.int_vars.clone(),
            };
            sub_parser.qregs.insert(
                "__q__".to_string(),
                Register {
                    offset: 0,
                    size: max_qubit,
                },
            );
            for (k, v) in saved_qregs {
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

            let instrs = sub_parser.parse_gate_application(expanded.trim(), line_num)?;
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
            param_vars: Some(float_vars),
            int_vars: Some(int_vars),
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
                Gate::Mcu(data) => Ok(Gate::mcu(data.mat, data.num_controls + 1)),
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

    fn split_gate_line(&self, line: &str, line_num: usize) -> Result<(String, Vec<f64>, String)> {
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
            let params =
                Self::parse_params_with_vars(params_str, line_num, self.param_vars.as_ref())?;
            let args_str = line[paren_end + 1..].trim().to_string();
            Ok((gate_name, params, args_str))
        } else {
            let first_space = line
                .find(char::is_whitespace)
                .ok_or_else(|| PrismError::Parse {
                    line: line_num,
                    message: format!("cannot parse instruction: `{line}`"),
                })?;
            let gate_name = line[..first_space].trim().to_string();
            let args_str = line[first_space..].trim().to_string();
            Ok((gate_name, vec![], args_str))
        }
    }

    fn parse_params_with_vars(
        params_str: &str,
        line_num: usize,
        vars: Option<&HashMap<String, f64>>,
    ) -> Result<Vec<f64>> {
        split_top_level_commas(params_str)
            .iter()
            .map(|p| eval_expr(p.trim(), line_num, vars))
            .collect()
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

    /// Handle gates that decompose into multiple instructions at parse time.
    ///
    /// Returns `Ok(None)` if the gate name is not a decomposed gate (caller
    /// should fall through to `resolve_gate`). Returns `Ok(Some(instrs))` for
    /// gates that expand to multiple instructions.
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
                Ok(Some(vec![Instruction::Gate {
                    gate: Gate::mcu(Gate::X.matrix_2x2(), controls as u8),
                    targets: SmallVec::from_slice(qubits),
                }]))
            }
            "rccx" => {
                Self::check_arity(name, qubits, 3)?;
                let c0 = qubits[0];
                let c1 = qubits[1];
                let target = qubits[2];
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::T,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c1, target],
                    },
                    Instruction::Gate {
                        gate: Gate::Tdg,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c0, target],
                    },
                    Instruction::Gate {
                        gate: Gate::T,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c1, target],
                    },
                    Instruction::Gate {
                        gate: Gate::Tdg,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![target],
                    },
                ]))
            }
            "rc3x" | "rcccx" => {
                Self::check_arity(name, qubits, 4)?;
                let c0 = qubits[0];
                let c1 = qubits[1];
                let c2 = qubits[2];
                let target = qubits[3];
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::T,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c2, target],
                    },
                    Instruction::Gate {
                        gate: Gate::Tdg,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c0, target],
                    },
                    Instruction::Gate {
                        gate: Gate::T,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c1, target],
                    },
                    Instruction::Gate {
                        gate: Gate::Tdg,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c0, target],
                    },
                    Instruction::Gate {
                        gate: Gate::T,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c1, target],
                    },
                    Instruction::Gate {
                        gate: Gate::Tdg,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::T,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![c2, target],
                    },
                    Instruction::Gate {
                        gate: Gate::Tdg,
                        targets: smallvec![target],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![target],
                    },
                ]))
            }
            "cswap" | "fredkin" => {
                Self::check_arity(name, qubits, 3)?;
                let ctrl = qubits[0];
                let t1 = qubits[1];
                let t2 = qubits[2];
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![t2, t1],
                    },
                    Instruction::Gate {
                        gate: Gate::mcu(Gate::X.matrix_2x2(), 2),
                        targets: smallvec![ctrl, t1, t2],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![t2, t1],
                    },
                ]))
            }
            "rxx" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q0, q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Rz(params[0]),
                        targets: smallvec![q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q0, q1],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q1],
                    },
                ]))
            }
            "ryy" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                let half_pi = std::f64::consts::FRAC_PI_2;
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::Rx(half_pi),
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::Rx(half_pi),
                        targets: smallvec![q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q0, q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Rz(params[0]),
                        targets: smallvec![q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q0, q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Rx(-half_pi),
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::Rx(-half_pi),
                        targets: smallvec![q1],
                    },
                ]))
            }
            "ecr" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::Rz(std::f64::consts::FRAC_PI_4),
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::Rx(std::f64::consts::FRAC_PI_2),
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q0, q1],
                    },
                    Instruction::Gate {
                        gate: Gate::X,
                        targets: smallvec![q0],
                    },
                ]))
            }
            "iswap" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::S,
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::S,
                        targets: smallvec![q1],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q0],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q0, q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q1, q0],
                    },
                    Instruction::Gate {
                        gate: Gate::H,
                        targets: smallvec![q1],
                    },
                ]))
            }
            "dcx" => {
                Self::check_arity(name, qubits, 2)?;
                let q0 = qubits[0];
                let q1 = qubits[1];
                Ok(Some(vec![
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q0, q1],
                    },
                    Instruction::Gate {
                        gate: Gate::Cx,
                        targets: smallvec![q1, q0],
                    },
                ]))
            }
            "u1" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                Ok(Some(vec![Instruction::Gate {
                    gate: Gate::P(params[0]),
                    targets: smallvec![qubits[0]],
                }]))
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
                Ok(Some(vec![Instruction::Gate {
                    gate: Gate::Fused(Box::new(mat)),
                    targets: smallvec![qubits[0]],
                }]))
            }
            "u3" | "u" | "U" => {
                Self::expect_param_count(name, params, 3, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                let theta = params[0];
                let phi = params[1];
                let lam = params[2];
                let mat = Self::u_matrix(theta, phi, lam);
                Ok(Some(vec![Instruction::Gate {
                    gate: Gate::Fused(Box::new(mat)),
                    targets: smallvec![qubits[0]],
                }]))
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

    fn u_matrix(theta: f64, phi: f64, lam: f64) -> [[Complex64; 2]; 2] {
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

    fn cu_target_matrix(theta: f64, phi: f64, lam: f64, gamma: f64) -> [[Complex64; 2]; 2] {
        let phase = Complex64::from_polar(1.0, gamma);
        let u = Self::u_matrix(theta, phi, lam);
        [
            [phase * u[0][0], phase * u[0][1]],
            [phase * u[1][0], phase * u[1][1]],
        ]
    }

    fn xx_plus_yy_matrix(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
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

    fn xx_minus_yy_matrix(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
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

    fn ms_matrix(phi0: f64, phi1: f64, theta: f64) -> [[Complex64; 4]; 4] {
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

    fn syc_matrix() -> [[Complex64; 4]; 4] {
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

    fn sqrt_iswap_matrix(sign: f64) -> [[Complex64; 4]; 4] {
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
