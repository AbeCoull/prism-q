use crate::error::{PrismError, Result};
use std::collections::HashMap;

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

pub(super) fn eval_expr(
    s: &str,
    line_num: usize,
    vars: Option<&HashMap<String, f64>>,
) -> Result<f64> {
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

pub(super) fn replace_word(haystack: &str, needle: &str, replacement: &str) -> String {
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

pub(super) fn split_top_level_commas(s: &str) -> Vec<&str> {
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
