//! Token stream for OpenQASM source: the one place that decides where a
//! comment, a string, a number or an operator ends.

use crate::error::{PrismError, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Ident,
    Int,
    Float,
    /// A number with its time unit attached, `10ns` or `4dt`. The token text
    /// keeps the unit.
    Duration,
    /// A double-quoted literal. The token text is the content without quotes.
    Str,
    /// `$0`, a hardware qubit. The token text is the digits.
    Physical,
    /// A whole `#pragma` or `pragma` line, text included, which the pragma
    /// grammar reads for itself.
    Pragma,
    /// The braced body of a `cal` or `defcal`, written in a calibration grammar
    /// this crate does not read. The token text is the content inside the braces.
    Calibration,
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
    Amp,
    AndAnd,
    Pipe,
    OrOr,
    Tilde,
    Shl,
    Shr,
    Eof,
}

impl Kind {
    /// How the token reads in an error message.
    pub(crate) fn describe(self) -> &'static str {
        match self {
            Kind::Ident => "a name",
            Kind::Int | Kind::Float => "a number",
            Kind::Duration => "a duration",
            Kind::Str => "a string",
            Kind::Physical => "a physical qubit",
            Kind::Pragma => "a pragma",
            Kind::Calibration => "a calibration block",
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
            Kind::Amp => "`&`",
            Kind::AndAnd => "`&&`",
            Kind::Pipe => "`|`",
            Kind::OrOr => "`||`",
            Kind::Tilde => "`~`",
            Kind::Shl => "`<<`",
            Kind::Shr => "`>>`",
            Kind::Eof => "the end of the program",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Token<'a> {
    pub kind: Kind,
    pub text: &'a str,
    pub line: u32,
    pub column: u32,
}

impl<'a> Token<'a> {
    /// The token as it should appear in an error message: its own text where
    /// that is informative, its kind where it is not.
    pub(crate) fn describe(&self) -> String {
        match self.kind {
            Kind::Ident | Kind::Int | Kind::Float | Kind::Duration => format!("`{}`", self.text),
            Kind::Physical => format!("`${}`", self.text),
            other => other.describe().to_string(),
        }
    }
}

/// Time units a number may carry, written straight after its digits. `s` comes
/// last so it never shadows a longer unit, and the micro sign has both of its
/// Unicode spellings beside `us`.
pub(crate) const TIME_UNITS: [&str; 7] = ["dt", "ns", "us", "\u{b5}s", "\u{3bc}s", "ms", "s"];

struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    at: usize,
    line: usize,
    line_start: usize,
    /// Set by a `cal` or `defcal` keyword, so the next `{` opens a calibration
    /// block rather than a scope.
    calibration_pending: bool,
}

/// Split `source` into tokens, ending with one [`Kind::Eof`].
///
/// Comments, whitespace and annotations are dropped here, so nothing downstream
/// rescans for them. A pragma line survives whole, since the pragma grammar is
/// its own, and so does a calibration body.
pub(crate) fn tokenize(source: &str) -> Result<Vec<Token<'_>>> {
    let mut lexer = Lexer {
        source,
        bytes: source.as_bytes(),
        at: 0,
        line: 1,
        line_start: 0,
        calibration_pending: false,
    };
    // Dense gate lists run near one token per 1.5 bytes (`cx q[0], q[1];`), so
    // half the byte count leaves one doubling at most on any real program.
    let mut out = Vec::with_capacity(source.len() / 2 + 1);
    loop {
        lexer.skip_trivia()?;
        if lexer.at >= lexer.bytes.len() {
            out.push(lexer.token(Kind::Eof, lexer.at, lexer.at));
            return Ok(out);
        }
        match lexer.bytes[lexer.at] {
            // An annotation is an `@` glued to a name where a statement starts,
            // which a modifier's `@` never is. It runs to the end of its line
            // and carries nothing an ideal simulation acts on.
            b'@' if statement_starts(&out) && lexer.ident_starts(lexer.at + 1) => {
                lexer.skip_line();
                continue;
            }
            b'{' if lexer.calibration_pending => {
                lexer.calibration_pending = false;
                out.push(lexer.calibration_block()?);
                continue;
            }
            _ => {}
        }
        let token = lexer.next_token()?;
        if token.kind == Kind::Ident
            && matches!(token.text, "cal" | "defcal")
            && statement_starts(&out)
        {
            lexer.calibration_pending = true;
        }
        out.push(token);
    }
}

/// True when the next token opens a statement.
fn statement_starts(out: &[Token]) -> bool {
    out.last().is_none_or(|token| {
        matches!(
            token.kind,
            Kind::Semicolon | Kind::LBrace | Kind::RBrace | Kind::Pragma | Kind::Calibration
        )
    })
}

impl<'a> Lexer<'a> {
    fn token(&self, kind: Kind, start: usize, end: usize) -> Token<'a> {
        Token {
            kind,
            text: &self.source[start..end],
            line: self.line as u32,
            column: (start - self.line_start + 1) as u32,
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
            return Ok(self.ident(start));
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
            b'<' if next == Some(b'<') => (Kind::Shl, 2),
            b'<' => (Kind::Lt, 1),
            b'>' if next == Some(b'=') => (Kind::Ge, 2),
            b'>' if next == Some(b'>') => (Kind::Shr, 2),
            b'>' => (Kind::Gt, 1),
            b'&' if next == Some(b'&') => (Kind::AndAnd, 2),
            b'&' => (Kind::Amp, 1),
            b'|' if next == Some(b'|') => (Kind::OrOr, 2),
            b'|' => (Kind::Pipe, 1),
            b'~' => (Kind::Tilde, 1),
            _ => return self.unicode_ident(start),
        };
        self.at += width;
        Ok(self.token(kind, start, self.at))
    }

    /// A name that opens with a non-ASCII letter, `θ` or `π`.
    fn unicode_ident(&mut self, start: usize) -> Result<Token<'a>> {
        let character = self.source[start..].chars().next().unwrap_or('?');
        if !character.is_alphabetic() {
            return Err(self.error(start, format!("unexpected character `{character}`")));
        }
        Ok(self.ident(start))
    }

    /// A name: letters, digits and `_`, where a letter may be any alphabetic
    /// Unicode character. `pragma` takes the rest of its line with it.
    fn ident(&mut self, start: usize) -> Token<'a> {
        loop {
            while self.at < self.bytes.len() && is_ident_byte(self.bytes[self.at]) {
                self.at += 1;
            }
            if self.bytes.get(self.at).is_none_or(u8::is_ascii) {
                break;
            }
            match self.source[self.at..].chars().next() {
                Some(character) if character.is_alphabetic() => {
                    self.at += character.len_utf8();
                }
                _ => break,
            }
        }
        if &self.source[start..self.at] == "pragma" {
            self.skip_line();
            return self.token(Kind::Pragma, start, self.at);
        }
        self.token(Kind::Ident, start, self.at)
    }

    /// True when a name could open at byte `at`.
    fn ident_starts(&self, at: usize) -> bool {
        match self.bytes.get(at) {
            Some(byte) if byte.is_ascii() => byte.is_ascii_alphabetic() || *byte == b'_',
            Some(_) => self.source[at..]
                .chars()
                .next()
                .is_some_and(char::is_alphabetic),
            None => false,
        }
    }

    fn skip_line(&mut self) {
        while self.at < self.bytes.len() && self.bytes[self.at] != b'\n' {
            self.at += 1;
        }
    }

    /// The braces of a `cal` or `defcal` body and everything between them, read
    /// as opaque text. Nested braces balance, and a string or comment may hold
    /// either brace without closing the block.
    fn calibration_block(&mut self) -> Result<Token<'a>> {
        let open = self.at;
        let opened = self.line;
        let column = (open - self.line_start + 1) as u32;
        self.at += 1;
        let mut depth = 1usize;
        while depth > 0 {
            let Some(&byte) = self.bytes.get(self.at) else {
                return Err(PrismError::Parse {
                    line: opened,
                    message: "unterminated calibration block".into(),
                });
            };
            match byte {
                b'\n' => {
                    self.newline();
                    continue;
                }
                b'{' => depth += 1,
                b'}' => depth -= 1,
                b'"' => {
                    self.at += 1;
                    while self.bytes.get(self.at).is_some_and(|byte| *byte != b'"') {
                        if self.bytes[self.at] == b'\n' {
                            self.newline();
                        } else {
                            self.at += 1;
                        }
                    }
                }
                b'/' if matches!(self.bytes.get(self.at + 1), Some(b'/' | b'*')) => {
                    self.skip_trivia()?;
                    continue;
                }
                _ => {}
            }
            self.at += 1;
        }
        Ok(Token {
            kind: Kind::Calibration,
            text: &self.source[open + 1..self.at - 1],
            line: opened as u32,
            column,
        })
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
        if let Some(width) = self.time_unit_width() {
            self.at += width;
            return Ok(self.token(Kind::Duration, start, self.at));
        }
        let kind = if float { Kind::Float } else { Kind::Int };
        Ok(self.token(kind, start, self.at))
    }

    /// Byte width of a time unit glued to the number just read, or `None`.
    ///
    /// A unit followed by more name characters is no unit, so `2sx` stays a
    /// number beside a name, as it lexed before durations existed.
    fn time_unit_width(&self) -> Option<usize> {
        let rest = &self.bytes[self.at..];
        if !rest
            .first()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || !byte.is_ascii())
        {
            return None;
        }
        let unit = TIME_UNITS
            .iter()
            .find(|unit| rest.starts_with(unit.as_bytes()))?;
        if rest
            .get(unit.len())
            .is_some_and(|byte| is_ident_byte(*byte))
        {
            return None;
        }
        Some(unit.len())
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
