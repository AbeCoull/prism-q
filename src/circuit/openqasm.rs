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

use super::expr::{eval_expr, replace_word, split_top_level_commas};

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

            if line.starts_with("reset") {
                instructions.extend(self.parse_reset(line, line_num)?);
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
#[path = "openqasm_tests.rs"]
mod tests;
