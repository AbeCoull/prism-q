//! Resolution of qubit and bit operands to indices.

use super::*;

impl<'a> Parser<'a> {
    /// Qubits an operand names: a physical index, a whole register, one
    /// element, a slice, or an alias of any of those.
    pub(in crate::circuit::openqasm) fn qubits_of(
        &self,
        operand: &Operand,
    ) -> Result<SmallVec<[usize; 4]>> {
        let name = match &operand.name {
            OperandName::Physical(index) => return Ok(smallvec![*index]),
            OperandName::Register(name) => name,
        };
        if let Some(indices) = self.alias_indices(ast::RegisterKind::Qubit, name, operand)? {
            return Ok(indices);
        }
        self.register_indices(&self.qregs, ast::RegisterKind::Qubit, name, operand)
    }

    pub(in crate::circuit::openqasm) fn bits_of(
        &self,
        operand: &Operand,
    ) -> Result<SmallVec<[usize; 4]>> {
        let Some(name) = operand.register() else {
            return Err(parse_error(
                operand.line,
                format!(
                    "`{}` is a qubit where a classical bit belongs",
                    operand.describe()
                ),
            ));
        };
        if let Some(indices) = self.alias_indices(ast::RegisterKind::Classical, name, operand)? {
            return Ok(indices);
        }
        self.register_indices(&self.cregs, ast::RegisterKind::Classical, name, operand)
    }

    pub(in crate::circuit::openqasm) fn bit_of(&self, operand: &Operand) -> Result<usize> {
        match self.bits_of(operand)?.as_slice() {
            [single] => Ok(*single),
            other => Err(parse_error(
                operand.line,
                format!(
                    "`{}` names {} bits where one belongs",
                    operand.describe(),
                    other.len()
                ),
            )),
        }
    }

    fn alias_indices(
        &self,
        kind: ast::RegisterKind,
        name: &str,
        operand: &Operand,
    ) -> Result<Option<SmallVec<[usize; 4]>>> {
        let Some(alias) = self.alias(name) else {
            return Ok(None);
        };
        if alias.kind != kind {
            return Err(parse_error(
                operand.line,
                format!("alias `{name}` names {}s", alias.kind.name()),
            ));
        }
        let Some(index) = &operand.index else {
            return Ok(Some(alias.indices.iter().copied().collect()));
        };
        let picks = self.pick_indices(alias.indices.len(), index, kind, operand)?;
        Ok(Some(picks.iter().map(|at| alias.indices[*at]).collect()))
    }

    fn register_indices(
        &self,
        registers: &HashMap<&'a str, Register>,
        kind: ast::RegisterKind,
        name: &str,
        operand: &Operand,
    ) -> Result<SmallVec<[usize; 4]>> {
        let register = registers
            .get(name)
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: name.to_string(),
                line: operand.line,
            })?;
        let Some(index) = &operand.index else {
            return Ok((0..register.size).map(|at| register.offset + at).collect());
        };
        let picks = self.pick_indices(register.size, index, kind, operand)?;
        Ok(picks.iter().map(|at| register.offset + at).collect())
    }

    /// Positions a subscript names on a run of `size`.
    pub(in crate::circuit::openqasm) fn pick_indices(
        &self,
        size: usize,
        index: &Index,
        kind: ast::RegisterKind,
        operand: &Operand,
    ) -> Result<SmallVec<[usize; 4]>> {
        let line = operand.line;
        let from_end = |value: i64| {
            if value < 0 {
                value + size as i64
            } else {
                value
            }
        };
        let bound = |value: i64| -> Result<usize> {
            let at = usize::try_from(from_end(value)).map_err(|_| {
                parse_error(
                    line,
                    format!(
                        "index {value} in `{}` is before its start",
                        operand.describe()
                    ),
                )
            })?;
            if at >= size {
                return Err(invalid_index(kind, at, size));
            }
            Ok(at)
        };
        match index {
            Index::Single(expr) => Ok(smallvec![bound(self.integer_of(expr, line)?)?]),
            Index::Set(entries) => {
                let mut out = SmallVec::new();
                for entry in entries {
                    out.push(bound(self.integer_of(entry, line)?)?);
                }
                Ok(out)
            }
            Index::Range(range) => {
                let at = |bound: &Option<Expr>, fallback: i64| -> Result<i64> {
                    match bound {
                        None => Ok(fallback),
                        Some(expr) => self.integer_of(expr, line),
                    }
                };
                let first = from_end(at(&range.start, 0)?);
                let last = from_end(at(&range.stop, size as i64 - 1)?);
                let stride = at(&range.step, 1)?;
                if stride == 0 {
                    return Err(parse_error(
                        line,
                        format!("range step in `{}` must be non-zero", operand.describe()),
                    ));
                }
                let mut out = SmallVec::new();
                let mut at = first;
                while (stride > 0 && at <= last) || (stride < 0 && at >= last) {
                    out.push(bound(at)?);
                    // A step past the integer range is past `last` too.
                    let Some(next) = at.checked_add(stride) else {
                        break;
                    };
                    at = next;
                }
                if out.is_empty() {
                    return Err(parse_error(
                        line,
                        format!("`{}` names no index", operand.describe()),
                    ));
                }
                Ok(out)
            }
        }
    }
}
