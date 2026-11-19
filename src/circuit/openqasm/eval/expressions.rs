//! Evaluation of classical expressions.

use super::*;

impl<'a> Parser<'a> {
    pub(in crate::circuit::openqasm) fn value_of(&self, expr: &Expr, line: usize) -> Result<f64> {
        Ok(self.typed_of(expr, line)?.to_float())
    }

    pub(in crate::circuit::openqasm) fn typed_of(
        &self,
        expr: &Expr,
        line: usize,
    ) -> Result<ClassicalValue> {
        self.reject_runtime(expr, line)?;
        self.reject_bit_value(expr, line)?;
        self.reject_symbol(expr, line)?;
        let expr = self.with_complex(expr)?;
        let expr = self.fold_externs(&expr, line)?;
        let vars = self.param_vars.as_ref();
        if !self.classical.is_empty() && expr.reads_element() {
            let resolved = replace_nodes(&expr, &|node| {
                let Expr::Element(element) = node else {
                    return Ok(None);
                };
                let Some(bits) = self.int_bits(element, line)? else {
                    return Ok(None);
                };
                let value = vars.and_then(|vars| vars.get(element.array));
                Ok(value.map(|value| {
                    let word = value.to_int();
                    let packed = bits
                        .iter()
                        .enumerate()
                        .fold(0i128, |acc, (at, &bit)| acc | (((word >> bit) & 1) << at));
                    Expr::Number(ClassicalValue::Int(packed))
                }))
            })?;
            return number_of(&resolved, line, vars, &self.durations, &self.arrays);
        }
        let value = number_of(&expr, line, vars, &self.durations, &self.arrays);
        if value.is_err() && expr.has_durationof() {
            let expr = self.with_durationof(&expr, line)?;
            return number_of(&expr, line, vars, &self.durations, &self.arrays);
        }
        value
    }

    /// Reject a classical bit read as a value under `parse`, where a measured bit
    /// reaches only a guard.
    fn reject_bit_value(&self, expr: &Expr, line: usize) -> Result<()> {
        if self.dynamic.is_some() || (self.cregs.is_empty() && self.params.is_empty()) {
            return Ok(());
        }
        let bit_leaf = |leaf: &Expr| {
            let name = match leaf {
                Expr::Ident(name) => *name,
                Expr::Element(element) => element.array,
                _ => return Ok(None),
            };
            let shadowed = self.classical.contains_key(name)
                || self.arrays.contains_key(name)
                || self.durations.contains_key(name)
                || self.complex.contains_key(name)
                || self.inputs.contains_key(name)
                || self.symbolic.contains(&name)
                || self
                    .param_vars
                    .as_ref()
                    .is_some_and(|vars| vars.contains_key(name));
            Ok((!shadowed && self.is_bit_name(name))
                .then_some(ClassicalExpr::Const(ClassicalValue::Bool(false))))
        };
        if super::dynamic::reads_leaf(expr, &bit_leaf)? {
            return Err(unsupported(
                line,
                format!(
                    "`{expr}` reads a classical bit as a value, which only a guard reads under \
                     `parse`; `openqasm::parse_dynamic` evaluates it at runtime"
                ),
            ));
        }
        Ok(())
    }

    /// The bits `x[i]` or `x[a:b]` reads of an `int[n]` or `uint[n]` `x`, the
    /// lowest bit of the value it reads as first; `None` for any other element.
    fn int_bits(
        &self,
        element: &syntax_expr::Element,
        line: usize,
    ) -> Result<Option<SmallVec<[u32; 4]>>> {
        use crate::circuit::dynamic::ClassicalType as Ty;
        if let [Expr::Range(range)] = element.indices.as_slice()
            && !self.arrays.contains_key(element.array)
            && let Some(ClassicalDecl {
                ty: ClassicalType::Int(Ty::Int { width } | Ty::Uint { width }),
                ..
            }) = self.classical.get(element.array)
        {
            let picks = range_positions(range, *width as usize, element.array, line, |index| {
                self.typed_of(index, line)
            })?;
            return Ok(Some(picks.into_iter().map(|at| at as u32).collect()));
        }
        Ok(self.int_bit(element, line)?.map(|at| smallvec![at]))
    }

    /// The bit `int[n] x` or `uint[n] x` reads as `x[i]`, `None` for any other
    /// element. A negative index counts from the top bit.
    pub(in crate::circuit::openqasm) fn int_bit(
        &self,
        element: &syntax_expr::Element,
        line: usize,
    ) -> Result<Option<u32>> {
        use crate::circuit::dynamic::ClassicalType as Ty;
        if self.arrays.contains_key(element.array) {
            return Ok(None);
        }
        let Some(ClassicalDecl {
            ty: ClassicalType::Int(Ty::Int { width } | Ty::Uint { width }),
            ..
        }) = self.classical.get(element.array)
        else {
            return Ok(None);
        };
        let name = element.array;
        let [index] = element.indices.as_slice() else {
            return Err(parse_error(
                line,
                format!("`{name}` is an integer and takes one bit index"),
            ));
        };
        let written = self.int_of(index, line)?;
        let at = if written < 0 {
            written + i128::from(*width)
        } else {
            written
        };
        if !(0..i128::from(*width)).contains(&at) {
            return Err(parse_error(
                line,
                format!("bit {written} is outside `{name}`, which holds {width} bits"),
            ));
        }
        Ok(Some(at as u32))
    }

    /// Fold `expr` to an integer: an integer exactly, a float only when whole.
    pub(super) fn int_of(&self, expr: &Expr, line: usize) -> Result<i128> {
        match self.typed_of(expr, line)? {
            ClassicalValue::Float(value) if !value.is_finite() || value.fract() != 0.0 => Err(
                parse_error(line, format!("expected an integer, got {value}")),
            ),
            value => Ok(value.to_int()),
        }
    }

    pub(in crate::circuit::openqasm) fn integer_of(&self, expr: &Expr, line: usize) -> Result<i64> {
        i64::try_from(self.int_of(expr, line)?)
            .map_err(|_| parse_error(line, "integer expression out of range"))
    }
}
