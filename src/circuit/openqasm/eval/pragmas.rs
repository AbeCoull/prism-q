//! Pragma statements.

use super::*;

impl<'a> Parser<'a> {
    /// Dispatch a `#pragma braket ...` line.
    ///
    /// Only the Braket dialect reads these. Under any other the pragma is
    /// rejected rather than dropped, so a lost result request cannot pass
    /// unnoticed.
    pub(super) fn exec_pragma(&mut self, text: &str, line: usize) -> Result<Vec<Instruction>> {
        let text = text.strip_prefix('#').unwrap_or(text);
        let body = text.strip_prefix("pragma").unwrap_or(text).trim();
        // A pragma no pass here reads is ignored, as the language asks. One
        // naming `braket` is read, or declined, since dropping a noise or result
        // request this crate understands would change the answer.
        let Some(braket_body) = body
            .strip_prefix("braket")
            .filter(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
        else {
            return Ok(Vec::new());
        };
        if self.dialect != Dialect::Braket {
            return Err(PrismError::UnsupportedConstruct {
                construct: "`#pragma braket`, which needs `Dialect::Braket`".to_string(),
                line,
            });
        }
        if self.nested {
            return Err(PrismError::UnsupportedConstruct {
                construct: "`#pragma braket` inside a block body".to_string(),
                line,
            });
        }
        let braket_body = braket_body.trim();
        let (kind, rest) = match braket_body.split_once(char::is_whitespace) {
            Some((kind, rest)) => (kind, rest.trim()),
            None => (braket_body, ""),
        };
        match kind {
            "result" => {
                let total = self.total_qubits;
                let spec = braket::parse_result_pragma(rest, line, total, &|token| {
                    self.qubits_of_text(token, line)
                        .map(|targets| targets.to_vec())
                })?;
                self.results.push(spec);
                Ok(Vec::new())
            }
            "noise" => {
                let spec = braket::parse_noise_pragma(rest, line, &|token| {
                    self.qubits_of_text(token, line)
                        .map(|targets| targets.to_vec())
                })?;
                self.pending_noise = Some(spec);
                Ok(Vec::new())
            }
            "verbatim" => {
                self.verbatim_pending = true;
                Ok(Vec::new())
            }
            _ if braket_body.starts_with("unitary") => self.exec_unitary_pragma(braket_body, line),
            other => Err(PrismError::UnsupportedConstruct {
                construct: format!("`#pragma braket {other}`"),
                line,
            }),
        }
    }

    /// `#pragma braket unitary([[...]]) q[0]`, an inline gate matrix.
    fn exec_unitary_pragma(&mut self, body: &str, line: usize) -> Result<Vec<Instruction>> {
        let rest = body.trim_start_matches("unitary").trim();
        if !rest.starts_with('(') {
            return Err(parse_error(
                line,
                "`unitary` takes its matrix in parentheses",
            ));
        }
        let (matrix_text, targets_text) = braket::split_paren_body(rest, line)?;
        let matrix = braket::parse_matrix(matrix_text, line)?;
        let targets = self.qubits_of_text(targets_text, line)?;
        braket::unitary_instructions(&matrix, &targets, line)
    }

    /// Resolve a qubit reference written inside a pragma, whose body is its own
    /// grammar rather than part of the statement stream.
    pub(in crate::circuit::openqasm) fn qubits_of_text(
        &self,
        token: &str,
        line: usize,
    ) -> Result<SmallVec<[usize; 4]>> {
        let mut out = SmallVec::new();
        for operand in syntax::parse_operands(&lexer::tokenize(token)?)? {
            out.extend(self.qubits_of(&Operand { line, ..operand })?);
        }
        Ok(out)
    }
}
