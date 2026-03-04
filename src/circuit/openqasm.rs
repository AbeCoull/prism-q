//! OpenQASM 3.0 parser — v0 subset.
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
//! | 1-qubit gates | `h q[0]; x q[1];` | id, x, y, z, h, s, sdg, t, tdg |
//! | Parametric gates | `rx(pi/4) q[0];` | rx, ry, rz — arithmetic expressions with `pi`, math functions |
//! | 2-qubit gates | `cx q[0], q[1];` | cx/cnot, cz, swap |
//! | Gate modifiers | `inv @ h q[0];` | `inv @`, `ctrl @` (chainable), `pow(k) @` (integer k) |
//! | Measurement (OQ3) | `c[0] = measure q[0];` | Assignment syntax (primary) |
//! | Measurement (OQ2) | `measure q[0] -> c[0];` | Arrow syntax (compat) |
//! | Register broadcast | `h q;` / `cx q, r;` | Applies gate to all qubits in register |
//! | Conditional (OQ2) | `if(c==1) x q[0];` | Classical register equality |
//! | Conditional (OQ3) | `if (c[0]) x q[0];` | Single classical bit test |
//! | Gate definition | `gate rxx(t) a,b { ... }` | User-defined gates |
//! | Barrier | `barrier q[0], q[1];` | |
//! | Line comments | `// comment` | |
//!
//! # Unsupported constructs (return `PrismError::UnsupportedConstruct`)
//!
//! - `def` / `defcal` definitions
//! - `for` / `while` loops
//! - `ctrl @ swap` modifier form (use `cswap` or `fredkin` keyword instead)
//! - `pow(k) @` with non-integer k (fractional powers)
//! - Classical expressions and types beyond `bit`
//! - Subroutines, `extern`, `box`, `duration`, `stretch`
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

struct Parser<'a> {
    input: &'a str,
    qregs: HashMap<String, Register>,
    cregs: HashMap<String, Register>,
    gate_defs: HashMap<String, GateDefinition>,
    total_qubits: usize,
    total_cbits: usize,
    gate_expansion_depth: usize,
    param_vars: Option<HashMap<String, f64>>,
}

const MAX_GATE_EXPANSION_DEPTH: usize = 32;
const MAX_REGISTER_SIZE: usize = 64;

// ── Expression evaluator ────────────────────────────────────────────────
//
// Recursive descent parser for gate parameter expressions.
//
// Grammar:
//   expr    → term (('+' | '-') term)*
//   term    → unary (('*' | '/') unary)*
//   unary   → '-' unary | primary
//   primary → NUMBER | 'pi' | 'tau' | IDENT '(' expr ')' | '(' expr ')' | IDENT

struct ExprParser<'e> {
    chars: &'e [u8],
    pos: usize,
    line: usize,
    vars: Option<&'e HashMap<String, f64>>,
}

impl<'e> ExprParser<'e> {
    fn new(input: &'e str, line: usize, vars: Option<&'e HashMap<String, f64>>) -> Self {
        Self {
            chars: input.as_bytes(),
            pos: 0,
            line,
            vars,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn eat(&mut self, ch: u8) -> bool {
        self.skip_ws();
        if self.pos < self.chars.len() && self.chars[self.pos] == ch {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> Result<f64> {
        let mut left = self.parse_term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    left += self.parse_term()?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    left -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_term(&mut self) -> Result<f64> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    left *= self.parse_unary()?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let right = self.parse_unary()?;
                    if right == 0.0 {
                        return Err(PrismError::Parse {
                            line: self.line,
                            message: "division by zero in angle expression".to_string(),
                        });
                    }
                    left /= right;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<f64> {
        if self.eat(b'-') {
            Ok(-self.parse_unary()?)
        } else if self.eat(b'+') {
            self.parse_unary()
        } else {
            self.parse_primary()
        }
    }

    fn parse_number(&mut self) -> Result<f64> {
        let start = self.pos;
        while self.pos < self.chars.len()
            && (self.chars[self.pos].is_ascii_digit()
                || self.chars[self.pos] == b'.'
                || self.chars[self.pos] == b'e'
                || self.chars[self.pos] == b'E'
                || ((self.chars[self.pos] == b'+' || self.chars[self.pos] == b'-')
                    && self.pos > start
                    && (self.chars[self.pos - 1] == b'e' || self.chars[self.pos - 1] == b'E')))
        {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.chars[start..self.pos]).unwrap_or("");
        let val = s.parse::<f64>().map_err(|_| PrismError::Parse {
            line: self.line,
            message: format!("invalid number: `{s}`"),
        })?;
        if !val.is_finite() {
            return Err(PrismError::Parse {
                line: self.line,
                message: format!("value is not finite: `{s}`"),
            });
        }
        Ok(val)
    }

    fn parse_ident(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.chars.len()
            && (self.chars[self.pos].is_ascii_alphanumeric() || self.chars[self.pos] == b'_')
        {
            self.pos += 1;
        }
        String::from_utf8_lossy(&self.chars[start..self.pos]).to_string()
    }

    fn parse_primary(&mut self) -> Result<f64> {
        self.skip_ws();
        if self.pos >= self.chars.len() {
            return Err(PrismError::Parse {
                line: self.line,
                message: "unexpected end of expression".to_string(),
            });
        }

        let ch = self.chars[self.pos];

        if ch == b'(' {
            self.pos += 1;
            let val = self.parse_expr()?;
            if !self.eat(b')') {
                return Err(PrismError::Parse {
                    line: self.line,
                    message: "unmatched `(` in expression".to_string(),
                });
            }
            return Ok(val);
        }

        if ch.is_ascii_digit() || ch == b'.' {
            return self.parse_number();
        }

        if ch == 0xCF || ch == 0xCE {
            let remaining = &self.chars[self.pos..];
            if remaining.starts_with("π".as_bytes()) {
                self.pos += "π".len();
                return Ok(std::f64::consts::PI);
            }
            if remaining.starts_with("τ".as_bytes()) {
                self.pos += "τ".len();
                return Ok(std::f64::consts::TAU);
            }
        }

        if ch.is_ascii_alphabetic() || ch == b'_' {
            let ident = self.parse_ident();
            self.skip_ws();
            if self.pos < self.chars.len() && self.chars[self.pos] == b'(' {
                self.pos += 1;
                let arg = self.parse_expr()?;
                if !self.eat(b')') {
                    return Err(PrismError::Parse {
                        line: self.line,
                        message: format!("unmatched `(` after function `{ident}`"),
                    });
                }
                return self.apply_function(&ident, arg);
            }
            return self.resolve_const_or_var(&ident);
        }

        Err(PrismError::Parse {
            line: self.line,
            message: format!("unexpected character `{}` in expression", ch as char),
        })
    }

    fn apply_function(&self, name: &str, arg: f64) -> Result<f64> {
        let val = match name {
            "sin" => arg.sin(),
            "cos" => arg.cos(),
            "tan" => arg.tan(),
            "asin" => arg.asin(),
            "acos" => arg.acos(),
            "atan" => arg.atan(),
            "sqrt" => arg.sqrt(),
            "exp" => arg.exp(),
            "ln" => arg.ln(),
            "log2" => arg.log2(),
            "abs" => arg.abs(),
            "ceil" => arg.ceil(),
            "floor" => arg.floor(),
            _ => {
                return Err(PrismError::Parse {
                    line: self.line,
                    message: format!("unknown function `{name}` in expression"),
                })
            }
        };
        if !val.is_finite() {
            return Err(PrismError::Parse {
                line: self.line,
                message: format!("{name}({arg}) produced non-finite result"),
            });
        }
        Ok(val)
    }

    fn resolve_const_or_var(&self, name: &str) -> Result<f64> {
        match name {
            "pi" => return Ok(std::f64::consts::PI),
            "tau" => return Ok(std::f64::consts::TAU),
            "euler" | "e" => return Ok(std::f64::consts::E),
            _ => {}
        }
        if let Some(vars) = self.vars {
            if let Some(&val) = vars.get(name) {
                return Ok(val);
            }
        }
        Err(PrismError::Parse {
            line: self.line,
            message: format!("unknown identifier `{name}` in expression"),
        })
    }
}

fn eval_expr(s: &str, line_num: usize, vars: Option<&HashMap<String, f64>>) -> Result<f64> {
    let s = s.trim();
    if s.is_empty() {
        return Err(PrismError::Parse {
            line: line_num,
            message: "empty expression".to_string(),
        });
    }
    let mut parser = ExprParser::new(s, line_num, vars);
    let val = parser.parse_expr()?;
    parser.skip_ws();
    if parser.pos < parser.chars.len() {
        return Err(PrismError::Parse {
            line: line_num,
            message: format!(
                "unexpected trailing characters in expression: `{}`",
                &s[parser.pos..]
            ),
        });
    }
    Ok(val)
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
fn utf8_char_width(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

fn replace_word(haystack: &str, needle: &str, replacement: &str) -> String {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    let nlen = nb.len();
    let mut result = String::with_capacity(haystack.len());
    let mut i = 0;
    while i + nlen <= hb.len() {
        if &hb[i..i + nlen] == nb {
            let before_ok = i == 0 || !is_ident_char(hb[i - 1]);
            let after_ok = i + nlen >= hb.len() || !is_ident_char(hb[i + nlen]);
            if before_ok && after_ok {
                result.push_str(replacement);
                i += nlen;
                continue;
            }
        }
        let ch_len = utf8_char_width(hb[i]);
        result.push_str(&haystack[i..i + ch_len]);
        i += ch_len;
    }
    while i < hb.len() {
        let ch_len = utf8_char_width(hb[i]);
        result.push_str(&haystack[i..i + ch_len]);
        i += ch_len;
    }
    result
}

fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(&s[start..]);
    result
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            gate_defs: HashMap::new(),
            total_qubits: 0,
            total_cbits: 0,
            gate_expansion_depth: 0,
            param_vars: None,
        }
    }

    fn parse(mut self) -> Result<Circuit> {
        let mut instructions = Vec::new();
        let mut gate_def_buf: Option<(String, usize)> = None;

        for (line_idx, raw_line) in self.input.lines().enumerate() {
            let line_num = line_idx + 1;

            let line = match raw_line.find("//") {
                Some(pos) => &raw_line[..pos],
                None => raw_line,
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some((ref mut buf, _start_line)) = gate_def_buf {
                buf.push(' ');
                buf.push_str(line);
                if line.contains('}') {
                    let (name, start) = gate_def_buf.take().unwrap();
                    self.parse_gate_def(&name, start)?;
                }
                continue;
            }

            let line = line.strip_suffix(';').unwrap_or(line).trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with("OPENQASM") {
                continue;
            }
            if line.starts_with("include") {
                continue;
            }

            // OQ3: qubit[n] name;
            if line.starts_with("qubit") {
                self.parse_qubit_decl(line, line_num)?;
                continue;
            }
            // OQ3: bit[n] name;
            if line.starts_with("bit") && !line.starts_with("bits") {
                self.parse_bit_decl(line, line_num)?;
                continue;
            }
            // OQ2 compat: qreg name[n];
            if line.starts_with("qreg") {
                self.parse_qreg_legacy(line, line_num)?;
                continue;
            }
            // OQ2 compat: creg name[n];
            if line.starts_with("creg") {
                self.parse_creg_legacy(line, line_num)?;
                continue;
            }

            // OQ2 compat: measure q[0] -> c[0];
            if line.starts_with("measure") {
                instructions.extend(self.parse_measure_arrow(line, line_num)?);
                continue;
            }

            // OQ3: c[0] = measure q[0];
            if line.contains("= measure") || line.contains("=measure") {
                instructions.extend(self.parse_measure_assign(line, line_num)?);
                continue;
            }

            if line.starts_with("barrier") {
                instructions.push(self.parse_barrier(line, line_num)?);
                continue;
            }

            if line.starts_with("if") {
                instructions.extend(self.parse_if_statement(line, line_num)?);
                continue;
            }

            if line.starts_with("gate ") || line.starts_with("gate(") {
                if line.contains('}') {
                    self.parse_gate_def(line, line_num)?;
                } else {
                    gate_def_buf = Some((line.to_string(), line_num));
                }
                continue;
            }

            let first_word = line
                .split(|c: char| c.is_whitespace() || c == '(')
                .next()
                .unwrap_or(line);
            if matches!(
                first_word,
                "def" | "defcal" | "opaque" | "for" | "while" | "box" | "extern" | "return"
            ) {
                return Err(PrismError::UnsupportedConstruct {
                    construct: first_word.to_string(),
                    line: line_num,
                });
            }

            instructions.extend(self.parse_gate_application(line, line_num)?);
        }

        Ok(Circuit {
            num_qubits: self.total_qubits,
            num_classical_bits: self.total_cbits,
            instructions,
        })
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
            if size > MAX_REGISTER_SIZE {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!(
                        "qubit register size {size} exceeds maximum ({MAX_REGISTER_SIZE})"
                    ),
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
            if size > MAX_REGISTER_SIZE {
                return Err(PrismError::Parse {
                    line: line_num,
                    message: format!(
                        "bit register size {size} exceeds maximum ({MAX_REGISTER_SIZE})"
                    ),
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
        let (name, idx) = Self::parse_indexed_ref(token, line_num)?;
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
        let (name, idx) = Self::parse_indexed_ref(token, line_num)?;
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

    fn parse_indexed_ref(token: &str, line_num: usize) -> Result<(&str, usize)> {
        let bracket = token.find('[').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: format!("expected indexed reference (e.g. `q[0]`), got: `{token}`"),
        })?;
        let end = token.find(']').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: format!("expected `]` in reference: `{token}`"),
        })?;
        let name = token[..bracket].trim();
        let idx: usize = token[bracket + 1..end]
            .trim()
            .parse()
            .map_err(|_| PrismError::Parse {
                line: line_num,
                message: format!("invalid index in `{token}`"),
            })?;
        Ok((name, idx))
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

    fn parse_barrier(&self, line: &str, line_num: usize) -> Result<Instruction> {
        let rest = line.strip_prefix("barrier").unwrap().trim();
        let mut qubits = SmallVec::<[usize; 4]>::new();
        for token in rest.split(',') {
            qubits.extend(self.resolve_qubit_arg(token.trim(), line_num)?);
        }
        Ok(Instruction::Barrier { qubits })
    }

    /// Parse `if(creg==value) gate args` (OQ2) or `if (c[i]) gate args` (OQ3).
    fn parse_if_statement(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let rest = line.strip_prefix("if").unwrap().trim();
        let open = rest.find('(').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `(` after `if`".to_string(),
        })?;
        let close = rest.find(')').ok_or_else(|| PrismError::Parse {
            line: line_num,
            message: "expected `)` in `if` condition".to_string(),
        })?;
        let cond_str = rest[open + 1..close].trim();
        let body_str = rest[close + 1..].trim();

        if body_str.is_empty() {
            return Err(PrismError::Parse {
                line: line_num,
                message: "expected gate after `if(...)` condition".to_string(),
            });
        }

        let condition = if let Some(eq_pos) = cond_str.find("==") {
            let reg_name = cond_str[..eq_pos].trim();
            let value_str = cond_str[eq_pos + 2..].trim();
            let value: u64 = value_str.parse().map_err(|_| PrismError::Parse {
                line: line_num,
                message: format!("invalid integer in `if` condition: `{value_str}`"),
            })?;
            let reg = self
                .cregs
                .get(reg_name)
                .ok_or_else(|| PrismError::UndefinedRegister {
                    name: reg_name.to_string(),
                    line: line_num,
                })?;
            ClassicalCondition::RegisterEquals {
                offset: reg.offset,
                size: reg.size,
                value,
            }
        } else if cond_str.contains('[') {
            let bit = self.resolve_cbit(cond_str, line_num)?;
            ClassicalCondition::BitIsOne(bit)
        } else {
            return Err(PrismError::Parse {
                line: line_num,
                message: format!(
                    "expected `creg==value` or `c[i]` in `if` condition, got: `{cond_str}`"
                ),
            });
        };

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

    fn parse_gate_application(&self, line: &str, line_num: usize) -> Result<Vec<Instruction>> {
        let (modifiers, gate_line) = Self::strip_modifiers(line, line_num)?;
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
                total_qubits: max_qubit,
                total_cbits: 0,
                gate_expansion_depth: self.gate_expansion_depth + 1,
                param_vars: Some(var_map.clone()),
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

            let instrs = sub_parser.parse_gate_application(expanded.trim(), line_num)?;
            all_instrs.extend(instrs);
        }

        Ok(Some(all_instrs))
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
            "p" => {
                Self::expect_param_count(name, params, 1, line_num)?;
                Ok(Gate::P(params[0]))
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
            "ch" => Ok(Gate::cu(Gate::H.matrix_2x2())),
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
            "u3" | "u" => {
                Self::expect_param_count(name, params, 3, line_num)?;
                Self::check_arity(name, qubits, 1)?;
                let theta = params[0];
                let phi = params[1];
                let lam = params[2];
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                let mat = [
                    [Complex64::new(c, 0.0), -Complex64::from_polar(s, lam)],
                    [
                        Complex64::from_polar(s, phi),
                        Complex64::from_polar(c, phi + lam),
                    ],
                ];
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
mod tests {
    use super::*;

    #[test]
    fn test_oq3_minimal_circuit() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nh q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 1);
        assert_eq!(c.gate_count(), 1);
    }

    #[test]
    fn test_oq3_bell_circuit() {
        let qasm = r#"
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.num_classical_bits, 2);
        assert_eq!(c.gate_count(), 2);
        assert_eq!(c.instructions.len(), 4);
    }

    #[test]
    fn test_oq2_compat() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q[0] -> c[0];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.gate_count(), 2);
        assert_eq!(c.instructions.len(), 3);
    }

    #[test]
    fn test_parametric_gates() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(pi/4) q[0];\nry(1.5707) q[0];\nrz(2*pi) q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 3);

        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rx(theta) => {
                    assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
                }
                _ => panic!("expected Rx"),
            }
        }
    }

    #[test]
    fn test_unsupported_gate_def() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\ndef mygate(qubit q) { x q; }";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::UnsupportedConstruct { .. }));
    }

    #[test]
    fn test_undefined_register() {
        let qasm = "OPENQASM 3.0;\nh q[0];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::UndefinedRegister { .. }));
    }

    #[test]
    fn test_qubit_out_of_bounds() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[5];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::InvalidQubit { .. }));
    }

    #[test]
    fn test_gate_arity_mismatch() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\ncx q[0], q[1], q[2];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::GateArity { .. }));
    }

    #[test]
    fn test_multiple_registers() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] a;
            qubit[3] b;
            h a[0];
            h b[2];
            cx a[1], b[0];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 5);
        assert_eq!(c.gate_count(), 3);
    }

    #[test]
    fn test_comments_stripped() {
        let qasm = r#"
            OPENQASM 3.0; // version
            qubit[1] q; // one qubit
            // full line comment
            h q[0]; // hadamard
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
    }

    #[test]
    fn test_barrier() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nbarrier q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 1);
        assert!(matches!(c.instructions[0], Instruction::Barrier { .. }));
    }

    #[test]
    fn test_oq3_measure_assign() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nx q[0];\nc[0] = measure q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        assert!(matches!(
            c.instructions[1],
            Instruction::Measure {
                qubit: 0,
                classical_bit: 0
            }
        ));
    }

    #[test]
    fn test_single_qubit_decl() {
        let qasm = "OPENQASM 3.0;\nqubit q;\nh q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 1);
        assert_eq!(c.gate_count(), 1);
    }

    #[test]
    fn test_inv_self_inverse() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\ninv @ h q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::H);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_inv_t_becomes_tdg() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\ninv @ t q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::Tdg);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_inv_parametric() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\ninv @ rx(pi/4) q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rx(theta) => {
                    assert!((theta + std::f64::consts::FRAC_PI_4).abs() < 1e-12);
                }
                _ => panic!("expected Rx"),
            }
        }
    }

    #[test]
    fn test_ctrl_x_becomes_cx() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ x q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(*gate, Gate::Cx);
            assert_eq!(targets.as_slice(), &[0, 1]);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_ctrl_z_becomes_cz() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ z q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::Cz);
        }
    }

    #[test]
    fn test_ctrl_h_becomes_cu() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ h q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert!(matches!(gate, Gate::Cu(_)));
            assert_eq!(gate.num_qubits(), 2);
            assert_eq!(targets.as_slice(), &[0, 1]);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_ctrl_parametric() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nctrl @ rz(pi/4) q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert!(matches!(gate, Gate::Cu(_)));
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_chained_inv_ctrl() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ninv @ ctrl @ rx(pi/4) q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert!(matches!(gate, Gate::Cu(_)));
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_pow_integer() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\npow(2) @ t q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            // T^2 = S
            let expected = Gate::S.matrix_2x2();
            if let Gate::Fused(mat) = gate {
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            (mat[i][j] - expected[i][j]).norm() < 1e-12,
                            "T^2 should be S"
                        );
                    }
                }
            } else {
                panic!("expected Fused, got {:?}", gate);
            }
        }
    }

    #[test]
    fn test_pow_zero_is_id() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\npow(0) @ x q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::Id);
        }
    }

    #[test]
    fn test_ctrl_ctrl_x_is_toffoli() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ ctrl @ x q[0], q[1], q[2];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(gate.num_qubits(), 3);
            assert_eq!(gate.name(), "mcu");
            assert_eq!(targets.as_slice(), &[0, 1, 2]);
        } else {
            panic!("expected gate instruction");
        }
    }

    #[test]
    fn test_ctrl_cx_is_toffoli() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ cx q[0], q[1], q[2];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(gate.num_qubits(), 3);
            assert_eq!(gate.name(), "mcu");
            assert_eq!(targets.as_slice(), &[0, 1, 2]);
        } else {
            panic!("expected gate instruction");
        }
    }

    #[test]
    fn test_ctrl_ctrl_ctrl_x() {
        let qasm = "OPENQASM 3.0;\nqubit[4] q;\nctrl @ ctrl @ ctrl @ x q[0], q[1], q[2], q[3];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(gate.num_qubits(), 4);
            assert_eq!(gate.name(), "mcu");
            assert_eq!(targets.as_slice(), &[0, 1, 2, 3]);
        } else {
            panic!("expected gate instruction");
        }
    }

    #[test]
    fn test_ctrl_swap_rejected() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ swap q[0], q[1], q[2];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::UnsupportedConstruct { .. }));
    }

    #[test]
    fn test_no_modifier_unchanged() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncx q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::Cx);
        }
    }

    #[test]
    fn test_cp_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncp(pi/4) q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(gate.num_qubits(), 2);
            assert_eq!(gate.name(), "cu");
            assert_eq!(targets.as_slice(), &[0, 1]);
            let phase = gate.controlled_phase().expect("should be CPhase");
            let expected = num_complex::Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
            assert!((phase - expected).norm() < 1e-12);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_cphase_alias() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncphase(pi) q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            let phase = gate.controlled_phase().unwrap();
            assert!((phase.re - (-1.0)).abs() < 1e-12);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_ctrl_cp_promotes_to_mcu() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nctrl @ cp(pi/4) q[0], q[1], q[2];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(gate.name(), "mcu");
            assert_eq!(targets.as_slice(), &[0, 1, 2]);
            assert!(gate.controlled_phase().is_some());
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_inv_cp() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ninv @ cp(pi/4) q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            let phase = gate.controlled_phase().unwrap();
            let expected = num_complex::Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
            assert!((phase - expected).norm() < 1e-12);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_cp_arity_mismatch() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\ncp(pi/4) q[0];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::GateArity { .. }));
    }

    #[test]
    fn test_sx_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nsx q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::SX);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_sxdg_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nsxdg q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::SXdg);
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_p_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(pi/4) q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::P(std::f64::consts::FRAC_PI_4));
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_cy_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncy q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(gate.num_qubits(), 2);
            assert_eq!(targets.as_slice(), &[0, 1]);
            if let Gate::Cu(mat) = gate {
                let expected = Gate::Y.matrix_2x2();
                for i in 0..2 {
                    for j in 0..2 {
                        assert!((mat[i][j] - expected[i][j]).norm() < 1e-12);
                    }
                }
            } else {
                panic!("expected Cu");
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_crx_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncrx(pi/2) q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            if let Gate::Cu(mat) = gate {
                let expected = Gate::Rx(std::f64::consts::FRAC_PI_2).matrix_2x2();
                for i in 0..2 {
                    for j in 0..2 {
                        assert!((mat[i][j] - expected[i][j]).norm() < 1e-12);
                    }
                }
            } else {
                panic!("expected Cu");
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_ccx_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nccx q[0], q[1], q[2];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(targets.as_slice(), &[0, 1, 2]);
            if let Gate::Mcu(data) = gate {
                assert_eq!(data.num_controls, 2);
                let x_mat = Gate::X.matrix_2x2();
                for (row_d, row_x) in data.mat.iter().zip(x_mat.iter()) {
                    for (d, x) in row_d.iter().zip(row_x.iter()) {
                        assert!((d - x).norm() < 1e-12);
                    }
                }
            } else {
                panic!("expected Mcu");
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_ccz_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nccz q[0], q[1], q[2];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            if let Gate::Mcu(data) = gate {
                assert_eq!(data.num_controls, 2);
                let z_mat = Gate::Z.matrix_2x2();
                for (row_d, row_z) in data.mat.iter().zip(z_mat.iter()) {
                    for (d, z) in row_d.iter().zip(row_z.iter()) {
                        assert!((d - z).norm() < 1e-12);
                    }
                }
            } else {
                panic!("expected Mcu");
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_cswap_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\ncswap q[0], q[1], q[2];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 3);
        assert!(
            matches!(&c.instructions[0], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [2, 1])
        );
        assert!(
            matches!(&c.instructions[1], Instruction::Gate { gate: Gate::Mcu(_), targets } if targets.as_slice() == [0, 1, 2])
        );
        assert!(
            matches!(&c.instructions[2], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [2, 1])
        );
    }

    #[test]
    fn test_rzz_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nrzz(pi/4) q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 1);
        assert!(matches!(
            &c.instructions[0],
            Instruction::Gate {
                gate: Gate::Rzz(_),
                ..
            }
        ));
    }

    #[test]
    fn test_rxx_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nrxx(pi/4) q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 7);
    }

    #[test]
    fn test_ryy_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nryy(pi/4) q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 7);
    }

    #[test]
    fn test_iswap_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\niswap q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 6);
    }

    #[test]
    fn test_ecr_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\necr q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 4);
    }

    #[test]
    fn test_dcx_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ndcx q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 2);
        assert!(
            matches!(&c.instructions[0], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [0, 1])
        );
        assert!(
            matches!(&c.instructions[1], Instruction::Gate { gate: Gate::Cx, targets } if targets.as_slice() == [1, 0])
        );
    }

    #[test]
    fn test_u1_is_p() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nu1(pi/4) q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::P(std::f64::consts::FRAC_PI_4));
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_u3_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nu3(pi/2, 0, pi) q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            if let Gate::Fused(mat) = gate {
                let h_mat = Gate::H.matrix_2x2();
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            (mat[i][j] - h_mat[i][j]).norm() < 1e-12,
                            "u3(pi/2, 0, pi) should match H: mat[{i}][{j}] = {:?} vs {:?}",
                            mat[i][j],
                            h_mat[i][j]
                        );
                    }
                }
            } else {
                panic!("expected Fused");
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_broadcast_1q_gate_on_register() {
        let qasm = "OPENQASM 3.0;\nqubit[4] q;\nh q;";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 4);
        assert_eq!(c.gate_count(), 4);
        for i in 0..4 {
            if let Instruction::Gate { gate, targets } = &c.instructions[i] {
                assert_eq!(*gate, Gate::H);
                assert_eq!(targets.as_slice(), &[i]);
            } else {
                panic!("expected gate at index {i}");
            }
        }
    }

    #[test]
    fn test_broadcast_2q_gate_pairwise() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            qubit[3] r;
            cx q, r;
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 6);
        assert_eq!(c.gate_count(), 3);
        for i in 0..3 {
            if let Instruction::Gate { gate, targets } = &c.instructions[i] {
                assert_eq!(*gate, Gate::Cx);
                assert_eq!(targets.as_slice(), &[i, i + 3]);
            } else {
                panic!("expected gate at index {i}");
            }
        }
    }

    #[test]
    fn test_broadcast_mixed_indexed_and_register() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            qubit[3] r;
            cx q[0], r;
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 3);
        for i in 0..3 {
            if let Instruction::Gate { gate, targets } = &c.instructions[i] {
                assert_eq!(*gate, Gate::Cx);
                assert_eq!(targets.as_slice(), &[0, i + 3]);
            } else {
                panic!("expected gate at index {i}");
            }
        }
    }

    #[test]
    fn test_broadcast_register_size_mismatch() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            qubit[3] r;
            cx q, r;
        "#;
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_broadcast_measure_arrow() {
        let qasm = r#"
            OPENQASM 2.0;
            qreg q[3];
            creg c[3];
            measure q -> c;
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 3);
        for i in 0..3 {
            assert!(matches!(
                c.instructions[i],
                Instruction::Measure {
                    qubit,
                    classical_bit
                } if qubit == i && classical_bit == i
            ));
        }
    }

    #[test]
    fn test_broadcast_measure_assign() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            bit[3] c;
            c = measure q;
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 3);
        for i in 0..3 {
            assert!(matches!(
                c.instructions[i],
                Instruction::Measure {
                    qubit,
                    classical_bit
                } if qubit == i && classical_bit == i
            ));
        }
    }

    #[test]
    fn test_broadcast_barrier() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nqubit[2] r;\nbarrier q, r;";
        let c = parse(qasm).unwrap();
        assert_eq!(c.instructions.len(), 1);
        if let Instruction::Barrier { qubits } = &c.instructions[0] {
            assert_eq!(qubits.as_slice(), &[0, 1, 2, 3, 4]);
        } else {
            panic!("expected barrier");
        }
    }

    #[test]
    fn test_broadcast_parametric_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[3] q;\nrz(pi/4) q;";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 3);
        for i in 0..3 {
            if let Instruction::Gate { gate, targets } = &c.instructions[i] {
                assert!(matches!(gate, Gate::Rz(_)));
                assert_eq!(targets.as_slice(), &[i]);
            } else {
                panic!("expected gate at index {i}");
            }
        }
    }

    #[test]
    fn test_broadcast_decomposed_gate() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            qubit[2] r;
            rzz(pi/4) q, r;
        "#;
        let c = parse(qasm).unwrap();
        // rzz emits 1 instruction per pair, 2 pairs
        assert_eq!(c.instructions.len(), 2);
    }

    #[test]
    fn test_if_oq2_register_equals() {
        let qasm = r#"
            OPENQASM 2.0;
            qreg q[2];
            creg c[2];
            if(c==1) x q[0];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Conditional {
            condition,
            gate,
            targets,
        } = &c.instructions[0]
        {
            assert_eq!(*gate, Gate::X);
            assert_eq!(targets.as_slice(), &[0]);
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterEquals {
                    offset: 0,
                    size: 2,
                    value: 1
                }
            ));
        } else {
            panic!("expected Conditional");
        }
    }

    #[test]
    fn test_if_oq3_single_bit() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            if (c[0]) x q[1];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Conditional {
            condition,
            gate,
            targets,
        } = &c.instructions[0]
        {
            assert_eq!(*gate, Gate::X);
            assert_eq!(targets.as_slice(), &[1]);
            assert!(matches!(condition, ClassicalCondition::BitIsOne(0)));
        } else {
            panic!("expected Conditional");
        }
    }

    #[test]
    fn test_if_with_parametric_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0]) rz(pi/4) q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Conditional { gate, .. } = &c.instructions[0] {
            assert!(matches!(gate, Gate::Rz(_)));
        } else {
            panic!("expected Conditional");
        }
    }

    #[test]
    fn test_if_missing_body() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0])";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_if_undefined_creg() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nif(c==0) x q[0];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::UndefinedRegister { .. }));
    }

    #[test]
    fn test_gate_def_no_params() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            gate bell a, b {
                h a;
                cx a, b;
            }
            bell q[0], q[1];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 2);
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(*gate, Gate::H);
            assert_eq!(targets.as_slice(), &[0]);
        } else {
            panic!("expected H gate");
        }
        if let Instruction::Gate { gate, targets } = &c.instructions[1] {
            assert_eq!(*gate, Gate::Cx);
            assert_eq!(targets.as_slice(), &[0, 1]);
        } else {
            panic!("expected CX gate");
        }
    }

    #[test]
    fn test_gate_def_with_params() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            gate myrz(theta) a {
                rz(theta) a;
            }
            myrz(pi/4) q[0];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rz(theta) => {
                    assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
                }
                _ => panic!("expected Rz"),
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_gate_def_single_line() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ngate mygate a, b { cx a, b; }\nmygate q[0], q[1];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            assert_eq!(*gate, Gate::Cx);
        } else {
            panic!("expected CX");
        }
    }

    #[test]
    fn test_gate_def_arity_mismatch() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            gate mygate a, b { cx a, b; }
            mygate q[0];
        "#;
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::GateArity { .. }));
    }

    #[test]
    fn test_gate_def_nested() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            gate myh a { h a; }
            gate mybell a, b { myh a; cx a, b; }
            mybell q[0], q[1];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 2);
    }

    #[test]
    fn test_missing_bracket_qubit_decl() {
        let qasm = "OPENQASM 3.0;\nqubit[4 q;";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_missing_bracket_bit_decl() {
        let qasm = "OPENQASM 3.0;\nbit[4 c;";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_pi_div_zero() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(pi/0) q[0];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_neg_pi_div_zero() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(-pi/0) q[0];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_nan_angle_rejected() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(NaN) q[0];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_inf_angle_rejected() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(inf) q[0];";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_register_size_too_large() {
        let qasm = "OPENQASM 3.0;\nqubit[999] q;";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_bit_register_size_too_large() {
        let qasm = "OPENQASM 3.0;\nbit[999] c;";
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_recursive_gate_def_rejected() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            gate loop a { loop a; }
            loop q[0];
        "#;
        let err = parse(qasm).unwrap_err();
        assert!(matches!(err, PrismError::Parse { .. }));
    }

    #[test]
    fn test_empty_gate_body() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            gate noop a { }
            noop q[0];
        "#;
        let err = parse(qasm);
        assert!(err.is_err(), "empty gate body should be rejected");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("empty body"),
            "error should mention empty body: {msg}"
        );
    }

    #[test]
    fn test_register_name_collision() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nqubit[2] q;";
        let err = parse(qasm);
        // Either error or silently overwrite — both acceptable, just don't panic
        assert!(err.is_ok() || err.is_err());
    }

    #[test]
    fn test_cry_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncry(pi/2) q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(targets.as_slice(), &[0, 1]);
            if let Gate::Cu(mat) = gate {
                let expected = Gate::Ry(std::f64::consts::FRAC_PI_2).matrix_2x2();
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            (mat[i][j] - expected[i][j]).norm() < 1e-12,
                            "CRy matrix mismatch at [{i}][{j}]"
                        );
                    }
                }
            } else {
                panic!("expected Cu, got {:?}", gate);
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_crz_gate() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\ncrz(pi/2) q[0], q[1];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, targets } = &c.instructions[0] {
            assert_eq!(targets.as_slice(), &[0, 1]);
            if let Gate::Cu(mat) = gate {
                let expected = Gate::Rz(std::f64::consts::FRAC_PI_2).matrix_2x2();
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            (mat[i][j] - expected[i][j]).norm() < 1e-12,
                            "CRz matrix mismatch at [{i}][{j}]"
                        );
                    }
                }
            } else {
                panic!("expected Cu, got {:?}", gate);
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_expr_literal_float() {
        assert!((eval_expr("1.234", 0, None).unwrap() - 1.234).abs() < 1e-12);
        assert!((eval_expr("0.25", 0, None).unwrap() - 0.25).abs() < 1e-12);
        assert!((eval_expr("-0.5", 0, None).unwrap() - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_expr_pi_constant() {
        let pi = std::f64::consts::PI;
        assert!((eval_expr("pi", 0, None).unwrap() - pi).abs() < 1e-12);
        assert!((eval_expr("-pi", 0, None).unwrap() - (-pi)).abs() < 1e-12);
        assert!((eval_expr("π", 0, None).unwrap() - pi).abs() < 1e-12);
    }

    #[test]
    fn test_expr_tau_constant() {
        assert!((eval_expr("tau", 0, None).unwrap() - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn test_expr_e_constant() {
        assert!((eval_expr("e", 0, None).unwrap() - std::f64::consts::E).abs() < 1e-12);
        assert!((eval_expr("euler", 0, None).unwrap() - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_expr_pi_division() {
        let pi = std::f64::consts::PI;
        assert!((eval_expr("pi/2", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
        assert!((eval_expr("pi/4", 0, None).unwrap() - pi / 4.0).abs() < 1e-12);
        assert!((eval_expr("-pi/2", 0, None).unwrap() - (-pi / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_expr_pi_multiplication() {
        let pi = std::f64::consts::PI;
        assert!((eval_expr("2*pi", 0, None).unwrap() - 2.0 * pi).abs() < 1e-12);
        assert!((eval_expr("0.5*pi", 0, None).unwrap() - 0.5 * pi).abs() < 1e-12);
    }

    #[test]
    fn test_expr_arithmetic() {
        assert!((eval_expr("1 + 2", 0, None).unwrap() - 3.0).abs() < 1e-12);
        assert!((eval_expr("5 - 3", 0, None).unwrap() - 2.0).abs() < 1e-12);
        assert!((eval_expr("2 * 3", 0, None).unwrap() - 6.0).abs() < 1e-12);
        assert!((eval_expr("6 / 2", 0, None).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_expr_operator_precedence() {
        assert!((eval_expr("2 + 3 * 4", 0, None).unwrap() - 14.0).abs() < 1e-12);
        assert!((eval_expr("2 * 3 + 4", 0, None).unwrap() - 10.0).abs() < 1e-12);
        assert!((eval_expr("10 - 2 * 3", 0, None).unwrap() - 4.0).abs() < 1e-12);
        assert!((eval_expr("10 / 2 + 3", 0, None).unwrap() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_expr_parentheses() {
        assert!((eval_expr("(2 + 3) * 4", 0, None).unwrap() - 20.0).abs() < 1e-12);
        assert!((eval_expr("2 * (3 + 4)", 0, None).unwrap() - 14.0).abs() < 1e-12);
        assert!((eval_expr("((1 + 2) * (3 + 4))", 0, None).unwrap() - 21.0).abs() < 1e-12);
    }

    #[test]
    fn test_expr_complex_pi_expressions() {
        let pi = std::f64::consts::PI;
        assert!((eval_expr("pi/2 + pi/4", 0, None).unwrap() - (pi / 2.0 + pi / 4.0)).abs() < 1e-12);
        assert!((eval_expr("2*pi/3", 0, None).unwrap() - (2.0 * pi / 3.0)).abs() < 1e-12);
        assert!((eval_expr("pi/2 + 0.1", 0, None).unwrap() - (pi / 2.0 + 0.1)).abs() < 1e-12);
        assert!((eval_expr("(pi + pi)/4", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_expr_unary_minus() {
        assert!((eval_expr("-1", 0, None).unwrap() - (-1.0)).abs() < 1e-12);
        assert!((eval_expr("-(2 + 3)", 0, None).unwrap() - (-5.0)).abs() < 1e-12);
        assert!((eval_expr("-(-1)", 0, None).unwrap() - 1.0).abs() < 1e-12);
        assert!((eval_expr("--1", 0, None).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_expr_unary_plus() {
        assert!((eval_expr("+1", 0, None).unwrap() - 1.0).abs() < 1e-12);
        assert!((eval_expr("+(2 + 3)", 0, None).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_expr_math_functions() {
        let pi = std::f64::consts::PI;
        assert!((eval_expr("sin(pi/2)", 0, None).unwrap() - 1.0).abs() < 1e-12);
        assert!((eval_expr("cos(0)", 0, None).unwrap() - 1.0).abs() < 1e-12);
        assert!((eval_expr("sqrt(4)", 0, None).unwrap() - 2.0).abs() < 1e-12);
        assert!((eval_expr("exp(0)", 0, None).unwrap() - 1.0).abs() < 1e-12);
        assert!((eval_expr("ln(1)", 0, None).unwrap() - 0.0).abs() < 1e-12);
        assert!((eval_expr("abs(-5)", 0, None).unwrap() - 5.0).abs() < 1e-12);
        assert!((eval_expr("asin(1)", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
        assert!((eval_expr("acos(1)", 0, None).unwrap() - 0.0).abs() < 1e-12);
        assert!((eval_expr("atan(0)", 0, None).unwrap() - 0.0).abs() < 1e-12);
        assert!((eval_expr("tan(0)", 0, None).unwrap() - 0.0).abs() < 1e-12);
        assert!((eval_expr("log2(8)", 0, None).unwrap() - 3.0).abs() < 1e-12);
        assert!((eval_expr("floor(2.7)", 0, None).unwrap() - 2.0).abs() < 1e-12);
        assert!((eval_expr("ceil(2.1)", 0, None).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_expr_nested_functions() {
        let pi = std::f64::consts::PI;
        assert!((eval_expr("sin(pi/4) * sin(pi/4)", 0, None).unwrap() - 0.5).abs() < 1e-12);
        assert!((eval_expr("sqrt(sin(pi/2))", 0, None).unwrap() - 1.0).abs() < 1e-12);
        assert!((eval_expr("asin(sin(pi/6))", 0, None).unwrap() - pi / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_expr_variables() {
        let mut vars = HashMap::new();
        vars.insert("theta".to_string(), std::f64::consts::FRAC_PI_4);
        vars.insert("phi".to_string(), std::f64::consts::FRAC_PI_2);
        let v = Some(&vars);
        assert!((eval_expr("theta", 0, v).unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!(
            (eval_expr("theta + phi", 0, v).unwrap()
                - (std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2))
                .abs()
                < 1e-12
        );
        assert!(
            (eval_expr("theta/2", 0, v).unwrap() - std::f64::consts::FRAC_PI_4 / 2.0).abs() < 1e-12
        );
        assert!(
            (eval_expr("2*theta + phi", 0, v).unwrap()
                - (2.0 * std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2))
                .abs()
                < 1e-12
        );
        assert!(
            (eval_expr("sin(theta)", 0, v).unwrap() - std::f64::consts::FRAC_PI_4.sin()).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_expr_division_by_zero() {
        assert!(eval_expr("1/0", 0, None).is_err());
        assert!(eval_expr("pi/0", 0, None).is_err());
    }

    #[test]
    fn test_expr_unknown_function() {
        assert!(eval_expr("foobar(1)", 0, None).is_err());
    }

    #[test]
    fn test_expr_unknown_variable() {
        assert!(eval_expr("xyz", 0, None).is_err());
    }

    #[test]
    fn test_expr_unbalanced_parens() {
        assert!(eval_expr("(1 + 2", 0, None).is_err());
        assert!(eval_expr("sin(pi/2", 0, None).is_err());
    }

    #[test]
    fn test_expr_empty() {
        assert!(eval_expr("", 0, None).is_err());
    }

    #[test]
    fn test_expr_trailing_chars() {
        assert!(eval_expr("1 2", 0, None).is_err());
    }

    #[test]
    fn test_expr_scientific_notation() {
        assert!((eval_expr("1e-3", 0, None).unwrap() - 0.001).abs() < 1e-15);
        assert!((eval_expr("2.5E2", 0, None).unwrap() - 250.0).abs() < 1e-12);
        assert!((eval_expr("1.5e+2", 0, None).unwrap() - 150.0).abs() < 1e-12);
    }

    #[test]
    fn test_qasm_arithmetic_param() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(pi/2 + pi/4) q[0];";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 1);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rx(theta) => {
                    let expected = std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_4;
                    assert!((theta - expected).abs() < 1e-12);
                }
                _ => panic!("expected Rx, got {:?}", gate),
            }
        } else {
            panic!("expected gate");
        }
    }

    #[test]
    fn test_qasm_multiply_pi() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(2*pi/3) q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rz(theta) => {
                    assert!((theta - 2.0 * std::f64::consts::PI / 3.0).abs() < 1e-12);
                }
                _ => panic!("expected Rz"),
            }
        }
    }

    #[test]
    fn test_qasm_function_in_param() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(sqrt(2)/2) q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::P(theta) => {
                    assert!((theta - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
                }
                _ => panic!("expected P"),
            }
        }
    }

    #[test]
    fn test_qasm_nested_function_in_param() {
        let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(sin(pi/4)) q[0];";
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::P(theta) => {
                    assert!((theta - (std::f64::consts::FRAC_PI_4).sin()).abs() < 1e-12);
                }
                _ => panic!("expected P"),
            }
        }
    }

    #[test]
    fn test_qasm_gate_def_with_expression_body() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            gate myrxx(theta) a, b {
                rx(theta/2) a;
                cx a, b;
                rx(-theta/2) a;
                cx a, b;
            }
            myrxx(pi) q[0], q[1];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 4);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rx(theta) => {
                    assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
                }
                _ => panic!("expected Rx, got {:?}", gate),
            }
        }
        if let Instruction::Gate { gate, .. } = &c.instructions[2] {
            match gate {
                Gate::Rx(theta) => {
                    assert!((theta - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
                }
                _ => panic!("expected Rx, got {:?}", gate),
            }
        }
    }

    #[test]
    fn test_qasm_gate_def_multi_param_expression() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            gate myu3(theta, phi, lambda) a {
                rz(lambda) a;
                ry(theta) a;
                rz(phi) a;
            }
            myu3(pi/2, pi/4, pi/8) q[0];
        "#;
        let c = parse(qasm).unwrap();
        assert_eq!(c.gate_count(), 3);
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rz(theta) => {
                    assert!((theta - std::f64::consts::PI / 8.0).abs() < 1e-12);
                }
                _ => panic!("expected Rz"),
            }
        }
        if let Instruction::Gate { gate, .. } = &c.instructions[1] {
            match gate {
                Gate::Ry(theta) => {
                    assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
                }
                _ => panic!("expected Ry"),
            }
        }
        if let Instruction::Gate { gate, .. } = &c.instructions[2] {
            match gate {
                Gate::Rz(theta) => {
                    assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
                }
                _ => panic!("expected Rz"),
            }
        }
    }

    #[test]
    fn test_qasm_gate_def_expression_with_arithmetic() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            gate half_rot(theta) a {
                rz(theta + pi) a;
            }
            half_rot(pi/4) q[0];
        "#;
        let c = parse(qasm).unwrap();
        if let Instruction::Gate { gate, .. } = &c.instructions[0] {
            match gate {
                Gate::Rz(theta) => {
                    let expected = std::f64::consts::FRAC_PI_4 + std::f64::consts::PI;
                    assert!((theta - expected).abs() < 1e-12);
                }
                _ => panic!("expected Rz"),
            }
        }
    }

    #[test]
    fn test_replace_word_boundary() {
        assert_eq!(replace_word("theta a", "a", "X"), "theta X");
        assert_eq!(replace_word("a theta a", "a", "X"), "X theta X");
        assert_eq!(replace_word("abc a ab", "a", "X"), "abc X ab");
        assert_eq!(
            replace_word("rz(theta) a", "a", "__q__[0]"),
            "rz(theta) __q__[0]"
        );
    }

    #[test]
    fn test_split_top_level_commas_nested() {
        let parts = split_top_level_commas("sin(pi/4), cos(0)");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].trim(), "sin(pi/4)");
        assert_eq!(parts[1].trim(), "cos(0)");
    }

    #[test]
    fn test_split_top_level_commas_simple() {
        let parts = split_top_level_commas("pi/2, pi/4, 0.1");
        assert_eq!(parts.len(), 3);
    }
}
