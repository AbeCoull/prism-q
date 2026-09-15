//! OpenQASM front end: source text to tokens, tokens to a syntax tree, and the
//! tree to instructions.
//!
//! The public entry points live in [`openqasm`](super::openqasm); this module
//! is what they are built on.

pub(crate) mod ast;
pub(crate) mod expr;
pub(crate) mod lexer;
pub(crate) mod parser;
pub(crate) mod stream;
