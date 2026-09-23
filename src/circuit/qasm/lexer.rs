//! Token stream for OpenQASM source: the one place that decides where a
//! comment, a string, a number or an operator ends.

use crate::error::{PrismError, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Ident,
    Int,
    Float,
    /// A double-quoted literal. The token text is the content without quotes.
    Str,
    /// `$0`, a hardware qubit. The token text is the digits.
    Physical,
    /// A whole `#pragma` line, text included, which the pragma grammar reads
    /// for itself.
    Pragma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Colon,
    At,
    Arrow,
    Concat,
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    EqEq,
    NotEq,
    Le,
    Ge,
    Lt,
    Gt,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Bang,
    Pow,
    Eof,
}

impl Kind {
    /// How the token reads in an error message.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Kind::Ident => "a name",
            Kind::Int | Kind::Float => "a number",
            Kind::Str => "a string",
            Kind::Physical => "a physical qubit",
            Kind::Pragma => "a pragma",
            Kind::LParen => "`(`",
            Kind::RParen => "`)`",
            Kind::LBracket => "`[`",
            Kind::RBracket => "`]`",
            Kind::LBrace => "`{`",
            Kind::RBrace => "`}`",
            Kind::Comma => "`,`",
            Kind::Semicolon => "`;`",
            Kind::Colon => "`:`",
            Kind::At => "`@`",
            Kind::Arrow => "`->`",
            Kind::Concat => "`++`",
            Kind::Assign => "`=`",
            Kind::AddAssign => "`+=`",
            Kind::SubAssign => "`-=`",
            Kind::MulAssign => "`*=`",
            Kind::DivAssign => "`/=`",
            Kind::ModAssign => "`%=`",
            Kind::EqEq => "`==`",
            Kind::NotEq => "`!=`",
            Kind::Le => "`<=`",
            Kind::Ge => "`>=`",
            Kind::Lt => "`<`",
            Kind::Gt => "`>`",
            Kind::Plus => "`+`",
            Kind::Minus => "`-`",
            Kind::Star => "`*`",
            Kind::Slash => "`/`",
            Kind::Percent => "`%`",
            Kind::Caret => "`^`",
            Kind::Bang => "`!`",
            Kind::Pow => "`**`",
            Kind::Eof => "the end of the program",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Token<'a> {
    pub kind: Kind,
    pub text: &'a str,
    pub line: usize,
    pub column: usize,
}

impl<'a> Token<'a> {
    /// The token as it should appear in an error message: its own text where
    /// that is informative, its kind where it is not.
    pub(crate) fn describe(&self) -> String {
        match self.kind {
            Kind::Ident | Kind::Int | Kind::Float => format!("`{}`", self.text),
            Kind::Physical => format!("`${}`", self.text),
            other => other.describe().to_string(),
        }
    }
}

/// The two Unicode constants the expression language accepts beside their
/// ASCII names.
const UNICODE_IDENTS: [&str; 2] = ["\u{3c0}", "\u{3c4}"];

struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    at: usize,
    line: usize,
    line_start: usize,
}

/// Split `source` into tokens, ending with one [`Kind::Eof`].
///
/// Comments and whitespace are dropped here, so nothing downstream rescans for
/// them. A `#pragma` line survives whole, since the pragma grammar is its own.
pub(crate) fn tokenize(source: &str) -> Result<Vec<Token<'_>>> {
    let mut lexer = Lexer {
        source,
        bytes: source.as_bytes(),
        at: 0,
        line: 1,
        line_start: 0,
    };
    // One token per four bytes is the ratio this grammar runs at, so the
    // stream is built in a single allocation for anything short of a pathology.
    let mut out = Vec::with_capacity(source.len() / 4 + 1);
    loop {
        lexer.skip_trivia()?;
        if lexer.at >= lexer.bytes.len() {
            out.push(lexer.token(Kind::Eof, lexer.at, lexer.at));
            return Ok(out);
        }
        out.push(lexer.next_token()?);
    }
}

impl<'a> Lexer<'a> {
    fn token(&self, kind: Kind, start: usize, end: usize) -> Token<'a> {
        Token {
            kind,
            text: &self.source[start..end],
            line: self.line,
            column: start - self.line_start + 1,
        }
    }

    fn newline(&mut self) {
        self.line += 1;
        self.at += 1;
        self.line_start = self.at;
    }

    fn skip_trivia(&mut self) -> Result<()> {
        while self.at < self.bytes.len() {
            match self.bytes[self.at] {
                b'\n' => self.newline(),
                byte if byte.is_ascii_whitespace() => self.at += 1,
                b'/' if self.bytes.get(self.at + 1) == Some(&b'/') => {
                    while self.at < self.bytes.len() && self.bytes[self.at] != b'\n' {
                        self.at += 1;
                    }
                }
                b'/' if self.bytes.get(self.at + 1) == Some(&b'*') => {
                    let opened = self.line;
                    self.at += 2;
                    loop {
                        if self.at >= self.bytes.len() {
                            return Err(PrismError::Parse {
                                line: opened,
                                message: "unterminated `/*` block comment".into(),
                            });
                        }
                        // Block comments do not nest, so the first `*/` closes
                        // the span whatever is inside it.
                        if self.bytes[self.at] == b'*' && self.bytes.get(self.at + 1) == Some(&b'/')
                        {
                            self.at += 2;
                            break;
                        }
                        if self.bytes[self.at] == b'\n' {
                            self.newline();
                        } else {
                            self.at += 1;
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
        Ok(())
    }

    fn next_token(&mut self) -> Result<Token<'a>> {
        let start = self.at;
        let byte = self.bytes[start];

        if byte == b'#' {
            while self.at < self.bytes.len() && self.bytes[self.at] != b'\n' {
                self.at += 1;
            }
            return Ok(self.token(Kind::Pragma, start, self.at));
        }

        if byte == b'"' {
            self.at += 1;
            while self.at < self.bytes.len() && self.bytes[self.at] != b'"' {
                if self.bytes[self.at] == b'\n' {
                    return Err(self.error(start, "unterminated string literal"));
                }
                // A backslash escapes the next byte, so a quoted path keeps
                // its separators.
                self.at += if self.bytes[self.at] == b'\\' { 2 } else { 1 };
            }
            if self.at >= self.bytes.len() {
                return Err(self.error(start, "unterminated string literal"));
            }
            let token = self.token(Kind::Str, start + 1, self.at);
            self.at += 1;
            return Ok(token);
        }

        if byte == b'$' {
            self.at += 1;
            let digits = self.at;
            while self.at < self.bytes.len() && self.bytes[self.at].is_ascii_digit() {
                self.at += 1;
            }
            if self.at == digits {
                return Err(self.error(start, "`$` needs a qubit index"));
            }
            return Ok(self.token(Kind::Physical, digits, self.at));
        }

        if byte.is_ascii_alphabetic() || byte == b'_' {
            while self.at < self.bytes.len() && is_ident_byte(self.bytes[self.at]) {
                self.at += 1;
            }
            return Ok(self.token(Kind::Ident, start, self.at));
        }

        if byte.is_ascii_digit() || (byte == b'.' && self.peek_digit(1)) {
            return self.number(start);
        }

        self.operator(start)
    }

    /// Punctuation, longest match first, so `**` never reads as two `*` and `++`
    /// never as two `+`.
    fn operator(&mut self, start: usize) -> Result<Token<'a>> {
        let next = self.bytes.get(start + 1).copied();
        let (kind, width) = match self.bytes[start] {
            b'(' => (Kind::LParen, 1),
            b')' => (Kind::RParen, 1),
            b'[' => (Kind::LBracket, 1),
            b']' => (Kind::RBracket, 1),
            b'{' => (Kind::LBrace, 1),
            b'}' => (Kind::RBrace, 1),
            b',' => (Kind::Comma, 1),
            b';' => (Kind::Semicolon, 1),
            b':' => (Kind::Colon, 1),
            b'@' => (Kind::At, 1),
            b'^' => (Kind::Caret, 1),
            b'*' if next == Some(b'*') => (Kind::Pow, 2),
            b'*' if next == Some(b'=') => (Kind::MulAssign, 2),
            b'*' => (Kind::Star, 1),
            b'+' if next == Some(b'+') => (Kind::Concat, 2),
            b'+' if next == Some(b'=') => (Kind::AddAssign, 2),
            b'+' => (Kind::Plus, 1),
            b'-' if next == Some(b'>') => (Kind::Arrow, 2),
            b'-' if next == Some(b'=') => (Kind::SubAssign, 2),
            b'-' => (Kind::Minus, 1),
            b'/' if next == Some(b'=') => (Kind::DivAssign, 2),
            b'/' => (Kind::Slash, 1),
            b'%' if next == Some(b'=') => (Kind::ModAssign, 2),
            b'%' => (Kind::Percent, 1),
            b'=' if next == Some(b'=') => (Kind::EqEq, 2),
            b'=' => (Kind::Assign, 1),
            b'!' if next == Some(b'=') => (Kind::NotEq, 2),
            b'!' => (Kind::Bang, 1),
            b'<' if next == Some(b'=') => (Kind::Le, 2),
            b'<' => (Kind::Lt, 1),
            b'>' if next == Some(b'=') => (Kind::Ge, 2),
            b'>' => (Kind::Gt, 1),
            _ => return self.unicode_ident(start),
        };
        self.at += width;
        Ok(self.token(kind, start, self.at))
    }

    /// `pi` and `tau` written as themselves, the only non-ASCII this grammar
    /// accepts.
    fn unicode_ident(&mut self, start: usize) -> Result<Token<'a>> {
        for name in UNICODE_IDENTS {
            if self.source[start..].starts_with(name) {
                self.at += name.len();
                return Ok(self.token(Kind::Ident, start, self.at));
            }
        }
        let character = self.source[start..].chars().next().unwrap_or('?');
        Err(self.error(start, format!("unexpected character `{character}`")))
    }

    fn peek_digit(&self, ahead: usize) -> bool {
        self.bytes
            .get(self.at + ahead)
            .is_some_and(u8::is_ascii_digit)
    }

    /// A numeric literal: a radix-prefixed integer, a decimal integer, or a
    /// float with an optional exponent. `_` separates digits in any of them.
    fn number(&mut self, start: usize) -> Result<Token<'a>> {
        if self.bytes[self.at] == b'0' {
            let radix = match self.bytes.get(self.at + 1) {
                Some(b'x' | b'X') => Some(16u32),
                Some(b'b' | b'B') => Some(2),
                Some(b'o' | b'O') => Some(8),
                _ => None,
            };
            if let Some(radix) = radix {
                self.at += 2;
                let digits = self.at;
                while self
                    .bytes
                    .get(self.at)
                    .is_some_and(|byte| *byte == b'_' || (*byte as char).is_digit(radix))
                {
                    self.at += 1;
                }
                if self.at == digits {
                    let prefix = &self.source[start..start + 2];
                    return Err(
                        self.error(start, format!("missing digits after `{prefix}` prefix"))
                    );
                }
                return Ok(self.token(Kind::Int, start, self.at));
            }
        }

        let mut float = false;
        while self.at < self.bytes.len() {
            match self.bytes[self.at] {
                byte if byte.is_ascii_digit() || byte == b'_' => self.at += 1,
                b'.' if !float => {
                    float = true;
                    self.at += 1;
                }
                b'e' | b'E' if self.exponent_follows() => {
                    float = true;
                    self.at += 2;
                    while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                        self.at += 1;
                    }
                }
                _ => break,
            }
        }
        let kind = if float { Kind::Float } else { Kind::Int };
        Ok(self.token(kind, start, self.at))
    }

    /// True when `e` here opens an exponent rather than an identifier, which
    /// is what keeps `2e` from swallowing a name that follows a number.
    fn exponent_follows(&self) -> bool {
        match self.bytes.get(self.at + 1) {
            Some(byte) if byte.is_ascii_digit() => true,
            Some(b'+' | b'-') => self.bytes.get(self.at + 2).is_some_and(u8::is_ascii_digit),
            _ => false,
        }
    }

    fn error(&self, start: usize, message: impl Into<String>) -> PrismError {
        let column = start - self.line_start + 1;
        PrismError::Parse {
            line: self.line,
            message: format!("{} at column {column}", message.into()),
        }
    }
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[cfg(test)]
#[path = "lexer_tests.rs"]
mod lexer_tests;
