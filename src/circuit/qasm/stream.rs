//! Cursor over a token slice, shared by the expression and statement parsers.

use super::lexer::{Kind, Token};
use crate::error::{PrismError, Result};

/// `'t` is the token slice, `'a` the source it was read from. They are
/// separate so a tree outlives the slice it was parsed from.
pub(crate) struct Stream<'t, 'a> {
    tokens: &'t [Token<'a>],
    at: usize,
}

impl<'t, 'a> Stream<'t, 'a> {
    pub(crate) fn new(tokens: &'t [Token<'a>]) -> Self {
        Self { tokens, at: 0 }
    }

    /// The token the cursor sits on. Never past the end: a stream always ends
    /// with [`Kind::Eof`], which every caller may read repeatedly.
    pub(crate) fn peek(&self) -> Token<'a> {
        self.tokens[self.at.min(self.tokens.len() - 1)]
    }

    pub(crate) fn peek_at(&self, ahead: usize) -> Token<'a> {
        self.tokens[(self.at + ahead).min(self.tokens.len() - 1)]
    }

    pub(crate) fn kind(&self) -> Kind {
        self.peek().kind
    }

    pub(crate) fn line(&self) -> usize {
        self.peek().line
    }

    pub(crate) fn at_end(&self) -> bool {
        self.kind() == Kind::Eof
    }

    pub(crate) fn advance(&mut self) -> Token<'a> {
        let token = self.peek();
        if token.kind != Kind::Eof {
            self.at += 1;
        }
        token
    }

    /// Consume the next token when it is `kind`, reporting whether it was.
    pub(crate) fn eat(&mut self, kind: Kind) -> bool {
        if self.kind() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume the next token when it is the keyword `word`.
    pub(crate) fn eat_keyword(&mut self, word: &str) -> bool {
        if self.is_keyword(word) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub(crate) fn is_keyword(&self, word: &str) -> bool {
        let token = self.peek();
        token.kind == Kind::Ident && token.text == word
    }

    pub(crate) fn expect(&mut self, kind: Kind) -> Result<Token<'a>> {
        if self.kind() == kind {
            return Ok(self.advance());
        }
        Err(self.expected(kind.describe()))
    }

    pub(crate) fn expect_ident(&mut self) -> Result<&'a str> {
        Ok(self.expect(Kind::Ident)?.text)
    }

    /// Error naming what the grammar wanted and what stood there instead.
    pub(crate) fn expected(&self, wanted: &str) -> PrismError {
        let token = self.peek();
        PrismError::Parse {
            line: token.line,
            message: format!(
                "expected {wanted}, found {} at column {}",
                token.describe(),
                token.column
            ),
        }
    }

    /// Position to restore with [`Stream::rewind`], for the one decision the
    /// grammar cannot make on a fixed lookahead.
    pub(crate) fn mark(&self) -> usize {
        self.at
    }

    pub(crate) fn rewind(&mut self, mark: usize) {
        self.at = mark;
    }
}
