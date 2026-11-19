//! Declarations of registers, classical variables, arrays, aliases and subroutines.

use super::*;

impl<'a> Parser<'a> {
    pub(super) fn declare_register(
        &mut self,
        kind: ast::RegisterKind,
        name: &'a str,
        size: Option<&Expr<'a>>,
        line: usize,
    ) -> Result<()> {
        let width = match size {
            None => 1usize,
            Some(expr) => {
                let value = self.integer_of(expr, line)?;
                if value <= 0 {
                    return Err(parse_error(
                        line,
                        format!("{} count must be > 0", kind.name()),
                    ));
                }
                value as usize
            }
        };
        self.reject_redeclaration(name, line)?;
        let grown = |total: usize| {
            total
                .checked_add(width)
                .filter(|&total| total <= MAX_DECLARED_WIDTH)
                .ok_or_else(|| too_wide(kind, line))
        };
        match kind {
            ast::RegisterKind::Qubit => {
                self.reject_mixed_addressing(line)?;
                let offset = self.total_qubits;
                self.total_qubits = grown(offset)?;
                Rc::make_mut(&mut self.qregs).insert(
                    name,
                    Register {
                        offset,
                        size: width,
                    },
                );
            }
            ast::RegisterKind::Classical => {
                let offset = self.total_cbits;
                self.total_cbits = grown(offset)?;
                Rc::make_mut(&mut self.cregs).insert(
                    name,
                    Register {
                        offset,
                        size: width,
                    },
                );
            }
        }
        Ok(())
    }

    /// Declare a `bit` a `def` body keeps to itself, on bits no declaration
    /// names. A runtime call writes its initial value on every call.
    pub(super) fn declare_private(
        &mut self,
        name: &'a str,
        size: Option<&Expr<'a>>,
        init: Option<&ast::BitInit<'a>>,
        line: usize,
    ) -> Result<()> {
        if self.dynamic.is_none() && self.in_outer_while {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`bit {name}` inside a def inlined into a `while` loop, whose passes would \
                     share it; declare it in the caller before the loop"
                ),
                line,
            });
        }
        let width = match size {
            None => 1usize,
            Some(expr) => match self.integer_of(expr, line)? {
                width if width > 0 => width as usize,
                _ => return Err(parse_error(line, "bit count must be > 0")),
            },
        };
        self.reject_redeclaration(name, line)?;
        let offset = self.private_register(width, line)?;
        Rc::make_mut(&mut self.cregs).insert(
            name,
            Register {
                offset,
                size: width,
            },
        );
        if self.dynamic.is_none() {
            if init.is_some() {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "an initializer on `{name}`, a bit an inlined def keeps to itself"
                    ),
                    line,
                });
            }
            return Ok(());
        }
        if let Some(init) = init {
            return self.init_bits(name, size, init, line);
        }
        let bits: Vec<usize> = (offset..offset + width).collect();
        for bits in bits.chunks(64) {
            self.write_classical_bits(bits, ClassicalExpr::Const(ClassicalValue::Int(0)), line)?;
        }
        Ok(())
    }

    /// The first of `width` bits taken from [`PRIVATE_BITS`] on.
    pub(in crate::circuit::openqasm) fn private_register(
        &self,
        width: usize,
        line: usize,
    ) -> Result<usize> {
        let taken = self.private_bits.as_ref().expect("a dynamic parse");
        let offset = taken.get();
        let total = offset
            .checked_add(width)
            .filter(|&total| total <= MAX_DECLARED_WIDTH)
            .ok_or_else(|| too_wide(ast::RegisterKind::Classical, line))?;
        taken.set(total);
        Ok(PRIVATE_BITS + offset)
    }

    /// Declare an `input`: a supplied one as a variable holding its value, an
    /// unsupplied `float` or `angle` as the next slot, read through a tag.
    pub(super) fn declare_input(
        &mut self,
        stmt: &Stmt<'a>,
        ty: &'a str,
        size: Option<&Expr<'a>>,
        name: &'a str,
        line: usize,
    ) -> Result<()> {
        if self.inputs.contains_key(name) {
            return Err(parse_error(line, format!("input `{name}` declared twice")));
        }
        let supplied = self
            .supplied
            .iter_mut()
            .find(|(supplied, _)| supplied == name)
            .and_then(|(_, value)| value.take());
        if let Some(value) = supplied {
            let value = Expr::Number(value);
            if ty == "bit" {
                self.declare_register(ast::RegisterKind::Classical, name, size, line)?;
                return self.init_bits(name, size, &ast::BitInit::Value(value), line);
            }
            if self.is_runtime_decl(stmt) {
                return self.declare_runtime(false, ty, size, name, Some(&value), line);
            }
            return self.declare_classical(false, ty, size, name, Some(&value), line);
        }
        if !matches!(ty, "float" | "angle") {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "input `{name}` is `{ty}`, which binds no gate angle; supply its value"
                ),
            });
        }
        if self.dynamic.is_some() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "unsupplied input `{name}` in a dynamic program, which has no slot to bind"
                ),
                line,
            });
        }
        let slot = self.input_names.len();
        self.inputs.insert(name, slot);
        self.input_names.push(name.to_string());
        let tag = self.tag(Linear {
            slot: Some(slot),
            scale: 1.0,
            offset: 0.0,
        });
        self.param_vars
            .get_or_insert_with(HashMap::new)
            .insert(name, ClassicalValue::Float(tag));
        self.symbolic.push(name);
        Ok(())
    }

    pub(in crate::circuit::openqasm) fn classical_type(
        &self,
        ty: &str,
        width: Option<&Expr>,
        line: usize,
    ) -> Result<ClassicalType> {
        Ok(match ty {
            "int" | "uint" => ClassicalType::Int(self.runtime_type(ty, width, line)?),
            "bool" => ClassicalType::Bool,
            "float" | "angle" => self.real_type(ty, width, line)?,
            "complex" => match self.real_type("float", width, line)? {
                ClassicalType::Float(bits) => ClassicalType::Complex(bits),
                other => other,
            },
            "duration" => ClassicalType::Duration,
            "stretch" => ClassicalType::Stretch,
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`{other}` declarations"),
                    line,
                });
            }
        })
    }

    pub(super) fn declare_classical(
        &mut self,
        constant: bool,
        ty: &str,
        width: Option<&Expr<'a>>,
        name: &'a str,
        value: Option<&Expr<'a>>,
        line: usize,
    ) -> Result<()> {
        let kind = self.classical_type(ty, width, line)?;
        self.reject_redeclaration(name, line)?;
        let Some(value) = value else {
            if constant {
                return Err(parse_error(
                    line,
                    format!("`const {ty} {name}` needs a value"),
                ));
            }
            if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
                let zero = Timed::Duration(Duration::default());
                self.bind_duration(name, kind, zero, constant, line)?;
            } else if let ClassicalType::Complex(_) = kind {
                self.bind_complex(name, kind, Complex64::new(0.0, 0.0), constant);
            } else {
                self.bind_classical(name, kind, ClassicalValue::Int(0), constant);
            }
            return Ok(());
        };
        if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
            let folded = self.duration_of(value, line)?;
            return self.bind_duration(name, kind, folded, constant, line);
        }
        if let ClassicalType::Complex(_) = kind {
            let folded = self.complex_of(value, line)?;
            self.bind_complex(name, kind, folded, constant);
            return Ok(());
        }
        let folded = self.fold_typed(kind, value, line)?;
        self.bind_classical(name, kind, folded, constant);
        Ok(())
    }

    /// Bind a `duration` or `stretch`. A stretch is its initializer, a lower
    /// bound, plus a growth of its own that a fixed-length `box` may resolve.
    fn bind_duration(
        &mut self,
        name: &'a str,
        ty: ClassicalType,
        value: Timed,
        constant: bool,
        line: usize,
    ) -> Result<()> {
        let value = match (ty, value.stretched()) {
            (ClassicalType::Stretch, Some(lower)) => {
                let id = self.stretch_ids.get();
                self.stretch_ids.set(id + 1);
                Timed::from(lower.plus(Stretched::growth(id), 1.0, line)?)
            }
            _ => value,
        };
        let decl = ClassicalDecl {
            ty,
            constant,
            value: constant.then_some(ConstValue::Duration(value)),
        };
        if Rc::make_mut(&mut self.classical)
            .insert(name, decl)
            .is_none()
        {
            self.declared.push(name);
        }
        self.durations.insert(name, value);
        Ok(())
    }

    /// Bind a `complex[float[n]]`, rounding each component to the declared width.
    fn bind_complex(&mut self, name: &'a str, ty: ClassicalType, value: Complex64, constant: bool) {
        let value = match ty {
            ClassicalType::Complex(width) => Complex64::new(
                crate::circuit::dynamic::real(value.re, width),
                crate::circuit::dynamic::real(value.im, width),
            ),
            _ => value,
        };
        let decl = ClassicalDecl {
            ty,
            constant,
            value: constant.then_some(ConstValue::Complex(value)),
        };
        if Rc::make_mut(&mut self.classical)
            .insert(name, decl)
            .is_none()
        {
            self.declared.push(name);
        }
        self.complex.insert(name, value);
    }

    /// Fold an expression to the complex value it has. A real one is promoted.
    pub(super) fn complex_of(&self, expr: &Expr, line: usize) -> Result<Complex64> {
        if self.reads_runtime(expr) {
            return Err(super::dynamic::runtime_complex(line));
        }
        if let Some(name) = self.symbol_in(expr) {
            return Err(self.symbol_error(name, "in a `complex` value", line));
        }
        let vars = self.param_vars.as_ref();
        let expr = self.with_complex(expr)?;
        let expr = if !self.arrays.is_empty() && expr.reads_element() {
            Cow::Owned(resolve_elements(
                &expr,
                line,
                vars,
                &self.durations,
                &self.arrays,
            )?)
        } else {
            expr
        };
        syntax_expr::eval_complex(&expr, line, vars)
    }

    /// `expr` with each `complex` name it reads replaced by its value.
    pub(super) fn with_complex<'x, 'e>(&self, expr: &'x Expr<'e>) -> Result<Cow<'x, Expr<'e>>> {
        if self.complex.is_empty() || !self.complex.keys().any(|name| expr.mentions(name)) {
            return Ok(Cow::Borrowed(expr));
        }
        replace_nodes(expr, &|node| {
            Ok(match node {
                Expr::Ident(name) => self.complex.get(name).map(|value| Expr::Complex(*value)),
                _ => None,
            })
        })
        .map(Cow::Owned)
    }

    /// Fold an expression that has to be a duration or a stretch.
    pub(in crate::circuit::openqasm) fn duration_of(
        &self,
        expr: &Expr,
        line: usize,
    ) -> Result<Timed> {
        self.reject_symbol(expr, line)?;
        let expr = self.with_durationof(expr, line)?;
        let vars = self.param_vars.as_ref();
        match timed_of(&expr, line, vars, &self.durations, &self.arrays)? {
            Timed::Number(value) => Err(parse_error(
                line,
                format!("`{expr}` is the number {value} where a duration belongs; give it a unit"),
            )),
            timed => Ok(timed),
        }
    }

    /// Check the length a `delay` or `box` names, returning it once no stretch
    /// is left in it.
    pub(in crate::circuit::openqasm) fn span_of(
        &self,
        expr: &Expr,
        what: &str,
        line: usize,
    ) -> Result<Option<Duration>> {
        let Timed::Duration(duration) = self.duration_of(expr, line)? else {
            return Ok(None);
        };
        if duration.is_negative() {
            return Err(parse_error(
                line,
                format!("`{what}` needs a non-negative duration, got `{expr}`"),
            ));
        }
        Ok(Some(duration))
    }

    pub(super) fn assign_classical(
        &mut self,
        target: &'a str,
        op: Option<AssignOp>,
        value: &Expr<'a>,
        line: usize,
    ) -> Result<()> {
        if let Some(var) = self.runtime_var(target) {
            return self.assign_runtime(var, op, value, line);
        }
        if self.arrays.contains_key(target) {
            let assign = ast::ElementAssign {
                array: target,
                indices: Vec::new(),
                op,
                value: value.clone(),
            };
            return self.assign_element(&assign, line);
        }
        let Some(decl) = self.classical.get(target) else {
            return Err(parse_error(
                line,
                format!("`{target}` is not a declared classical variable"),
            ));
        };
        let (kind, constant) = (decl.ty, decl.constant);
        if constant {
            return Err(parse_error(
                line,
                format!("`{target}` is `const` and cannot be assigned"),
            ));
        }
        self.reject_guarded_write(target, line)?;
        if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
            return self.assign_duration(target, kind, op, value, line);
        }
        if let ClassicalType::Complex(_) = kind {
            let folded = self.complex_of(value, line)?;
            let updated = match op {
                None => folded,
                Some(op) => {
                    let current = self.complex.get(target).copied().unwrap_or_default();
                    let combined = Expr::Binary {
                        op: op.binary(),
                        left: Box::new(Expr::Complex(current)),
                        right: Box::new(Expr::Complex(folded)),
                    };
                    syntax_expr::eval_complex(&combined, line, None)?
                }
            };
            self.bind_complex(target, kind, updated, false);
            return Ok(());
        }
        let folded = self.fold_typed(kind, value, line)?;
        let updated = match op {
            None => folded,
            Some(op) => {
                let current = self.typed_of(&Expr::Ident(target), line)?;
                syntax_expr::arithmetic(op.binary(), current, folded, line)?
            }
        };
        self.bind_classical(target, kind, updated, false);
        Ok(())
    }

    /// The element type of an `array[ty[width], ...]`.
    pub(super) fn array_kind(
        &self,
        ty: &str,
        width: Option<&Expr>,
        line: usize,
    ) -> Result<ClassicalType> {
        match ty {
            "int" | "uint" | "bool" | "float" | "angle" => self.classical_type(ty, width, line),
            other => Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`array[{other}, ...]`; an array holds `int`, `uint`, `float`, \
                     `angle` or `bool`, and bits belong in a `bit[n]` register"
                ),
                line,
            }),
        }
    }

    /// The array `expr` names, copied out: a whole array, a slice or a
    /// partial subscript of one, or arrays joined with `++` along their first
    /// dimension. `None` when `expr` is a single value.
    fn array_value(&self, expr: &Expr, line: usize) -> Result<Option<ArrayValue>> {
        with_stack(|| match expr {
            Expr::Ident(name) => Ok(self.arrays.get(name).map(|array| ArrayValue {
                dims: array.dims.clone(),
                values: array.values.clone(),
            })),
            Expr::Element(element) => {
                let Some(array) = self.arrays.get(element.array) else {
                    return Ok(None);
                };
                let view = array_view(array, element.array, &element.indices, line, |index| {
                    self.typed_of(index, line)
                })?;
                if view.dims.is_empty() {
                    return Ok(None);
                }
                Ok(Some(ArrayValue {
                    dims: view.dims,
                    values: view.offsets.iter().map(|&at| array.values[at]).collect(),
                }))
            }
            Expr::Concat(parts) => {
                let mut joined: Option<ArrayValue> = None;
                for part in parts {
                    let Some(next) = self.array_value(part, line)? else {
                        return Err(parse_error(
                            line,
                            format!("`{part}` is not an array, so `++` cannot join it"),
                        ));
                    };
                    match joined.as_mut() {
                        None => joined = Some(next),
                        Some(acc) if acc.dims[1..] == next.dims[1..] => {
                            acc.dims[0] += next.dims[0];
                            acc.values.extend(next.values);
                        }
                        Some(acc) => {
                            return Err(parse_error(
                                line,
                                format!(
                                    "`++` joins arrays of shapes {:?} and {:?}, whose inner \
                                     dimensions differ",
                                    acc.dims, next.dims
                                ),
                            ));
                        }
                    }
                }
                Ok(joined)
            }
            _ => Ok(None),
        })
    }

    pub(super) fn declare_array(&mut self, decl: &ast::ArrayDecl<'a>, line: usize) -> Result<()> {
        let kind = self.array_kind(decl.ty, decl.width.as_ref(), line)?;
        self.reject_redeclaration(decl.name, line)?;
        let mut dims = Vec::with_capacity(decl.dims.len());
        let mut total = 1usize;
        for dim in &decl.dims {
            let size = self.integer_of(dim, line)?;
            let size = usize::try_from(size)
                .ok()
                .filter(|&size| size > 0)
                .ok_or_else(|| parse_error(line, format!("array size must be > 0, got {size}")))?;
            total = total
                .checked_mul(size)
                .filter(|&total| total <= MAX_ARRAY_ELEMENTS)
                .ok_or_else(|| {
                    parse_error(
                        line,
                        format!(
                            "array `{}` holds more than {MAX_ARRAY_ELEMENTS} elements",
                            decl.name
                        ),
                    )
                })?;
            dims.push(size);
        }
        if decl.constant && decl.init.is_none() {
            return Err(parse_error(
                line,
                format!("`const array {}` needs a value", decl.name),
            ));
        }
        let mut values = Vec::with_capacity(total);
        match &decl.init {
            None => values.resize(total, kind.store(ClassicalValue::Int(0))),
            Some(init) => self.fold_initializer(kind, init, &dims, decl.name, line, &mut values)?,
        }
        let array = ClassicalArray {
            ty: kind,
            constant: decl.constant,
            dims,
            values,
        };
        if self.arrays.insert(decl.name, Rc::new(array)).is_none() {
            self.declared.push(decl.name);
        }
        Ok(())
    }

    /// Fold an initializer into `values` in row-major order, checking that each
    /// braced list has one entry per index of the dimension it stands at.
    fn fold_initializer(
        &self,
        kind: ClassicalType,
        init: &ast::ArrayInit,
        dims: &[usize],
        name: &str,
        line: usize,
        values: &mut Vec<ClassicalValue>,
    ) -> Result<()> {
        match (init, dims.split_first()) {
            (ast::ArrayInit::Value(expr), None) => {
                values.push(self.fold_typed(kind, expr, line)?);
                Ok(())
            }
            (ast::ArrayInit::Value(expr), Some(_)) => {
                let Some(source) = self.array_value(expr, line)? else {
                    return Err(parse_error(
                        line,
                        format!("initializer for `{name}` does not match its dimensions"),
                    ));
                };
                if source.dims != dims {
                    return Err(parse_error(
                        line,
                        format!(
                            "initializer `{expr}` for `{name}` has shape {:?} where a shape of \
                             {dims:?} stands",
                            source.dims
                        ),
                    ));
                }
                values.extend(source.values.into_iter().map(|value| kind.store(value)));
                Ok(())
            }
            (ast::ArrayInit::List(entries), Some((&size, inner))) if entries.len() == size => {
                for entry in entries {
                    self.fold_initializer(kind, entry, inner, name, line, values)?;
                }
                Ok(())
            }
            (ast::ArrayInit::List(entries), Some((&size, _))) => Err(parse_error(
                line,
                format!(
                    "initializer for `{name}` lists {} entries where a dimension of size \
                     {size} stands",
                    entries.len()
                ),
            )),
            _ => Err(parse_error(
                line,
                format!("initializer for `{name}` does not match its dimensions"),
            )),
        }
    }

    pub(super) fn assign_element(
        &mut self,
        assign: &ast::ElementAssign<'a>,
        line: usize,
    ) -> Result<()> {
        let Some(array) = self.arrays.get(assign.array) else {
            return Err(parse_error(
                line,
                format!(
                    "`{}` is not a name an assignment can write",
                    Expr::Element(Box::new(syntax_expr::Element {
                        array: assign.array,
                        indices: assign.indices.clone(),
                    }))
                ),
            ));
        };
        if array.constant {
            return Err(parse_error(
                line,
                format!("`{}` is `const` and cannot be assigned", assign.array),
            ));
        }
        self.reject_guarded_write(assign.array, line)?;
        if assign.indices.len() < array.dims.len()
            || assign
                .indices
                .iter()
                .any(|index| matches!(index, Expr::Range(_)))
        {
            return self.assign_slice(assign, line);
        }
        let at = element_offset(array, assign.array, &assign.indices, line, |index| {
            self.typed_of(index, line)
        })?;
        let kind = array.ty;
        let current = array.values[at];
        let folded = self.fold_typed(kind, &assign.value, line)?;
        let updated = kind.store(match assign.op {
            None => folded,
            Some(op) => syntax_expr::arithmetic(op.binary(), current, folded, line)?,
        });
        if let Some(array) = self.arrays.get_mut(assign.array) {
            Rc::make_mut(array).values[at] = updated;
        }
        Ok(())
    }

    /// Copy an array value into the elements a slice or a partial subscript
    /// names, which must have the shape the value has.
    fn assign_slice(&mut self, assign: &ast::ElementAssign<'a>, line: usize) -> Result<()> {
        let array = &self.arrays[assign.array];
        let kind = array.ty;
        let view = array_view(array, assign.array, &assign.indices, line, |index| {
            self.typed_of(index, line)
        })?;
        let target = if assign.indices.is_empty() {
            assign.array.to_string()
        } else {
            Expr::Element(Box::new(syntax_expr::Element {
                array: assign.array,
                indices: assign.indices.clone(),
            }))
            .to_string()
        };
        if assign.op.is_some() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("a compound assignment to the array slice `{target}`"),
                line,
            });
        }
        let Some(source) = self.array_value(&assign.value, line)? else {
            return Err(parse_error(
                line,
                format!(
                    "`{}` is not an array, so it cannot fill `{target}`, whose shape is {:?}",
                    assign.value, view.dims
                ),
            ));
        };
        if source.dims != view.dims {
            return Err(parse_error(
                line,
                format!(
                    "shape mismatch: `{target}` has shape {:?} and `{}` has shape {:?}",
                    view.dims, assign.value, source.dims
                ),
            ));
        }
        if let Some(array) = self.arrays.get_mut(assign.array) {
            let array = Rc::make_mut(array);
            for (&at, value) in view.offsets.iter().zip(source.values) {
                array.values[at] = kind.store(value);
            }
        }
        Ok(())
    }

    fn assign_duration(
        &mut self,
        target: &'a str,
        kind: ClassicalType,
        op: Option<AssignOp>,
        value: &Expr<'a>,
        line: usize,
    ) -> Result<()> {
        if kind == ClassicalType::Stretch {
            return Err(parse_error(
                line,
                format!(
                    "`{target}` is a stretch, which a scheduler sizes and a program cannot assign"
                ),
            ));
        }
        self.reject_symbol(value, line)?;
        let value = self.with_durationof(value, line)?;
        let value = timed_of(
            &value,
            line,
            self.param_vars.as_ref(),
            &self.durations,
            &self.arrays,
        )?;
        let updated = match op {
            None => value,
            Some(op) => {
                let current = self
                    .durations
                    .get(target)
                    .copied()
                    .unwrap_or(Timed::Duration(Duration::default()));
                syntax_expr::combine_timed(op.binary(), current, value, line)?
            }
        };
        if let Timed::Number(number) = updated {
            return Err(parse_error(
                line,
                format!("assignment leaves duration `{target}` holding the number {number}"),
            ));
        }
        self.bind_duration(target, kind, updated, false, line)
    }

    /// Fold an initializer to the value its declared type holds. An integer or
    /// `bool` takes a whole number, which a fractional float is not.
    pub(super) fn fold_typed(
        &self,
        kind: ClassicalType,
        value: &Expr,
        line: usize,
    ) -> Result<ClassicalValue> {
        if let Some(name) = self.symbol_in(value) {
            return Err(self.symbol_error(
                name,
                "in a classical value; an unsupplied input binds only a gate angle",
                line,
            ));
        }
        Ok(kind.store(match kind {
            ClassicalType::Int(_) | ClassicalType::Bool => {
                ClassicalValue::Int(self.int_of(value, line)?)
            }
            _ => self.typed_of(value, line)?,
        }))
    }

    pub(super) fn declare_alias(
        &mut self,
        name: &'a str,
        sources: &[Operand<'a>],
        line: usize,
    ) -> Result<()> {
        self.reject_redeclaration(name, line)?;
        let mut kind: Option<ast::RegisterKind> = None;
        let mut indices: Vec<usize> = Vec::new();
        for source in sources {
            let (source_kind, resolved) = self.alias_source(source, line)?;
            match kind {
                Some(seen) if seen != source_kind => {
                    return Err(parse_error(
                        line,
                        format!("alias `{name}` joins qubits and classical bits"),
                    ));
                }
                _ => kind = Some(source_kind),
            }
            indices.extend(resolved);
        }
        let kind =
            kind.ok_or_else(|| parse_error(line, format!("alias `{name}` names nothing")))?;
        Rc::make_mut(&mut self.aliases).insert(name, Alias { kind, indices });
        Ok(())
    }

    /// One operand of an alias, resolved on the side of the register wall the
    /// name it opens with sits on.
    fn alias_source(
        &self,
        source: &Operand,
        line: usize,
    ) -> Result<(ast::RegisterKind, SmallVec<[usize; 4]>)> {
        let Some(name) = source.register() else {
            return Ok((ast::RegisterKind::Qubit, self.qubits_of(source)?));
        };
        let alias_kind = self.alias(name).map(|alias| alias.kind);
        if self.qregs.contains_key(name) || alias_kind == Some(ast::RegisterKind::Qubit) {
            return Ok((ast::RegisterKind::Qubit, self.qubits_of(source)?));
        }
        if self.cregs.contains_key(name) || alias_kind == Some(ast::RegisterKind::Classical) {
            return Ok((ast::RegisterKind::Classical, self.bits_of(source)?));
        }
        Err(PrismError::UndefinedRegister {
            name: name.to_string(),
            line,
        })
    }

    pub(super) fn declare_def(
        &mut self,
        name: &'a str,
        args: &[DefParam<'a>],
        result: Option<&ast::DefType<'a>>,
        body: &Block<'a>,
        line: usize,
    ) -> Result<()> {
        self.reject_extern_name(name, line)?;
        let mut args = args.to_vec();
        // Freeze signature widths before a caller can shadow their bindings.
        for param in &mut args {
            if let DefParam::Array { width, shape, .. } = param {
                let sizes = match shape {
                    ArrayShape::Sized(dims) => dims.iter_mut().collect::<Vec<_>>(),
                    ArrayShape::Rank(rank) => vec![rank],
                };
                for size in sizes.into_iter().chain(width.as_mut()) {
                    let value = i128::from(self.integer_of(size, line)?);
                    *size = Expr::Number(ClassicalValue::Int(value));
                }
            }
            if let DefParam::Qubit {
                width: Some(width), ..
            }
            | DefParam::Bit {
                width: Some(width), ..
            }
            | DefParam::Value {
                width: Some(width), ..
            } = param
            {
                let value = i128::from(self.integer_of(width, line)?);
                *width = Expr::Number(ClassicalValue::Int(value));
            }
        }
        for param in &args {
            match param {
                DefParam::Value { ty, width, .. } => {
                    self.classical_type(ty, width.as_ref(), line)?;
                }
                DefParam::Array { ty, width, .. } => {
                    self.array_kind(ty, width.as_ref(), line)?;
                }
                _ => {}
            }
        }
        let mut result = result.cloned();
        if let Some(width) = result.as_mut().and_then(|result| result.width.as_mut()) {
            let value = i128::from(self.integer_of(width, line)?);
            *width = Expr::Number(ClassicalValue::Int(value));
        }
        if let Some(result) = result.as_ref().filter(|result| result.ty != "bit") {
            self.classical_type(result.ty, result.width.as_ref(), line)?;
        }
        self.charge(self.arrays.len(), line)?;
        let arrays = self
            .arrays
            .iter()
            .filter(|(_, array)| array.constant)
            .map(|(name, array)| (*name, Rc::clone(array)))
            .collect();
        let constants = Rc::clone(&self.classical);
        let Some((last, rest)) = body.split_last() else {
            if result.is_some() {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares a result but its body is empty"),
                ));
            }
            Rc::make_mut(&mut self.def_defs).insert(
                name,
                DefDefinition {
                    args,
                    result: None,
                    constants,
                    arrays,
                    local: None,
                    body: Block::new(),
                },
            );
            return Ok(());
        };
        let early = return_line(rest).or_else(|| match last.kind {
            StmtKind::Return(_) => None,
            _ => return_line(std::slice::from_ref(last)),
        });
        let computed_bit = self.dynamic.is_some()
            && matches!(&last.kind, StmtKind::Return(Some(ast::ReturnValue::Value(value)))
                if value.as_ident().is_none_or(|name| !rest.iter().any(|stmt| matches!(&stmt.kind,
                    StmtKind::RegisterDecl { kind: ast::RegisterKind::Classical, name: local, .. } if *local == name))));
        let local = if result.as_ref().is_some_and(|result| result.ty == "bit")
            && early.is_none()
            && !computed_bit
        {
            Self::def_result_local(name, &args, true, last, rest)?
        } else {
            Self::check_returns(name, result.as_ref(), body)?;
            if result.is_some() && return_line(body).is_none() {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares a result but does not end by returning one"),
                ));
            }
            None
        };
        let private = self.private_bits.is_some().then(Vec::new);
        Self::check_def_writes(name, &args, local, body, true, private)?;
        Rc::make_mut(&mut self.def_defs).insert(
            name,
            DefDefinition {
                args,
                result,
                constants,
                arrays,
                local,
                body: body.clone(),
            },
        );
        Ok(())
    }

    pub(super) fn reject_extern_name(&self, name: &str, line: usize) -> Result<()> {
        if self.externs.contains_key(name) {
            return Err(parse_error(line, format!("`{name}` is already an extern")));
        }
        Ok(())
    }

    fn check_returns(name: &str, result: Option<&ast::DefType>, body: &[Stmt]) -> Result<()> {
        with_stack(|| {
            for stmt in body {
                match &stmt.kind {
                    StmtKind::Return(value) if value.is_some() != result.is_some() => {
                        return Err(parse_error(
                            stmt.line,
                            format!(
                                "def `{name}` returns a value inconsistent with its declared result"
                            ),
                        ));
                    }
                    StmtKind::If(branch) => {
                        Self::check_returns(name, result, &branch.then_body)?;
                        if let Some(body) = &branch.else_body {
                            Self::check_returns(name, result, body)?;
                        }
                    }
                    StmtKind::For { body, .. }
                    | StmtKind::While { body, .. }
                    | StmtKind::Box { body, .. } => Self::check_returns(name, result, body)?,
                    StmtKind::Switch { arms, .. } => {
                        for arm in arms {
                            Self::check_returns(name, result, &arm.body)?;
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        })
    }

    pub(super) fn exec_return(
        &mut self,
        value: Option<&ast::ReturnValue<'a>>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let Some(frame) = &self.def_result else {
            return Err(PrismError::UnsupportedConstruct {
                construct: "`return` outside a def".to_string(),
                line,
            });
        };
        if self.guard.is_some() {
            return Err(PrismError::UnsupportedConstruct {
                construct: "a measurement-dependent return from a def".to_string(),
                line,
            });
        }
        if frame.runtime {
            return self.exec_runtime_return(value, line, out);
        }
        let ty = frame.ty;
        let returned = match (value, ty) {
            (Some(ast::ReturnValue::Measure(source)), ty) => {
                if ty.is_some() {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: "a measurement returned as a classical scalar".to_string(),
                        line,
                    });
                }
                let qubits = self.qubits_of(source)?;
                out.extend(Self::build_measurements(qubits, frame.bits.clone(), line)?);
                None
            }
            (Some(ast::ReturnValue::Value(value)), Some(ty)) => {
                let value = self.resolve_calls(value, line, out)?;
                if let ClassicalType::Complex(width) = ty {
                    let value = self.complex_of(&value, line)?;
                    let frame = self.def_result.as_mut().expect("checked above");
                    frame.value = Some(Expr::Complex(Complex64::new(
                        crate::circuit::dynamic::real(value.re, width),
                        crate::circuit::dynamic::real(value.im, width),
                    )));
                    frame.returned = true;
                    return Ok(());
                }
                // Folded once here, so the runtime check below does not call an
                // extern that the fold then calls again.
                let value = if self.reads_runtime(&value) {
                    value
                } else {
                    self.fold_externs(&value, line)?.into_owned()
                };
                if ty != ClassicalType::Duration
                    && self
                        .lower_expr(&value, line)
                        .is_ok_and(|value| !matches!(value, ClassicalExpr::Const(_)))
                {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: "a measurement-dependent scalar returned from a def".to_string(),
                        line,
                    });
                }
                Some(if ty == ClassicalType::Duration {
                    match self.duration_of(&value, line)? {
                        Timed::Duration(duration) => Expr::Duration(duration),
                        Timed::Stretch(_) => {
                            return Err(PrismError::UnsupportedConstruct {
                                construct: "a stretch returned as a duration".to_string(),
                                line,
                            });
                        }
                        Timed::Number(_) => unreachable!("duration_of rejects numbers"),
                    }
                } else {
                    Expr::Number(ty.store(self.fold_typed(ty, &value, line)?))
                })
            }
            (Some(ast::ReturnValue::Value(value)), None) if frame.local.is_none() => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("return of `{value}` as a bit without a classical move"),
                    line,
                });
            }
            _ => None,
        };
        let frame = self.def_result.as_mut().expect("checked above");
        frame.value = returned;
        frame.returned = true;
        Ok(())
    }

    /// Whether a def call in the running statement reached `end` under `parse`,
    /// so nothing after the call in that statement may run. `parse_dynamic`
    /// leaves the rest to the halt block it emitted.
    pub(super) fn unwinds_end(&self) -> bool {
        self.exit == Some(Exit::End) && self.dynamic.is_none()
    }

    /// The error that carries an `end` out of an expression; `execute_block` drops it.
    fn end_unwind() -> PrismError {
        parse_error(0, String::new())
    }

    pub(in crate::circuit::openqasm) fn resolve_calls(
        &mut self,
        expr: &Expr<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<Expr<'a>> {
        with_stack(|| {
            Ok(match expr {
                Expr::Call(call) => {
                    let args = call
                        .args
                        .iter()
                        .map(|arg| self.resolve_calls(arg, line, out))
                        .collect::<Result<Vec<_>>>()?;
                    if self.def_defs.contains_key(call.name) {
                        let args = args
                            .into_iter()
                            .map(|arg| match arg {
                                Expr::Element(element) if element.indices.len() == 1 => {
                                    Argument::Operand(Operand {
                                        name: OperandName::Register(element.array),
                                        index: Some(Index::Single(element.indices[0].clone())),
                                        line,
                                    })
                                }
                                value => Argument::Value(value),
                            })
                            .collect::<Vec<_>>();
                        if self.streaming() {
                            self.flush(std::mem::take(out));
                        }
                        let (instructions, value) =
                            self.expand_def(call.name, &args, None, line)?;
                        out.extend(instructions);
                        if self.unwinds_end() {
                            return Err(Self::end_unwind());
                        }
                        if self.streaming() {
                            self.flush(std::mem::take(out));
                        }
                        value.ok_or_else(|| {
                            parse_error(
                                line,
                                format!(
                                    "def `{}` has no scalar result for this expression",
                                    call.name
                                ),
                            )
                        })?
                    } else {
                        Expr::Call(Box::new(syntax_expr::Call {
                            name: call.name,
                            args,
                        }))
                    }
                }
                Expr::Binary { op, left, right } => {
                    let left = self.resolve_calls(left, line, out)?;
                    if matches!(op, syntax_expr::BinaryOp::And | syntax_expr::BinaryOp::Or) {
                        match self.typed_of(&left, line) {
                            Ok(value) if value.truthy() == (*op == syntax_expr::BinaryOp::Or) => {
                                return Ok(Expr::Number(ClassicalValue::Bool(value.truthy())));
                            }
                            Err(_)
                                if right
                                    .finds_call(&|name| self.def_defs.contains_key(name))
                                    .is_some() =>
                            {
                                return self.resolve_short_circuit(*op, &left, right, line, out);
                            }
                            _ => {}
                        }
                    }
                    Expr::Binary {
                        op: *op,
                        left: Box::new(left),
                        right: Box::new(self.resolve_calls(right, line, out)?),
                    }
                }
                Expr::Negate(inner) => {
                    Expr::Negate(Box::new(self.resolve_calls(inner, line, out)?))
                }
                Expr::Not(inner) => Expr::Not(Box::new(self.resolve_calls(inner, line, out)?)),
                Expr::BitNot(inner) => {
                    Expr::BitNot(Box::new(self.resolve_calls(inner, line, out)?))
                }
                Expr::Cast(cast) => Expr::Cast(Box::new(syntax_expr::Cast {
                    ty: cast.ty,
                    width: cast.width.clone(),
                    value: self.resolve_calls(&cast.value, line, out)?,
                })),
                Expr::Element(element) => Expr::Element(Box::new(syntax_expr::Element {
                    array: element.array,
                    indices: element
                        .indices
                        .iter()
                        .map(|index| self.resolve_calls(index, line, out))
                        .collect::<Result<_>>()?,
                })),
                _ => expr.clone(),
            })
        })
    }

    pub(super) fn resolve_arguments<'p>(
        &mut self,
        args: &'p [Argument<'a>],
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<Cow<'p, [Argument<'a>]>> {
        if self.def_defs.is_empty()
            || !args.iter().any(|arg| {
                matches!(
                    arg, Argument::Value(value)
                        if value.finds_call(&|name| self.def_defs.contains_key(name)).is_some()
                )
            })
        {
            return Ok(Cow::Borrowed(args));
        }
        args.iter()
            .map(|arg| {
                Ok(match arg {
                    Argument::Value(value) => {
                        Argument::Value(self.resolve_calls(value, line, out)?)
                    }
                    Argument::Operand(_) => arg.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Cow::Owned)
    }

    /// The bit a `def` body declares and returns by name, after checking that
    /// its final statement returns what the signature promises.
    fn def_result_local(
        name: &str,
        args: &[DefParam<'a>],
        has_result: bool,
        last: &Stmt<'a>,
        rest: &[Stmt<'a>],
    ) -> Result<Option<&'a str>> {
        let value = match &last.kind {
            StmtKind::Return(value) => value.as_ref(),
            _ if has_result => None,
            _ => return Ok(None),
        };
        let Some(value) = value else {
            if has_result {
                return Err(parse_error(
                    last.line,
                    format!(
                        "def `{name}` declares a `bit` result but does not end by returning one"
                    ),
                ));
            }
            return Ok(None);
        };
        if !has_result {
            return Err(parse_error(
                last.line,
                format!("def `{name}` returns a value but declares no `-> bit` result"),
            ));
        }
        let ast::ReturnValue::Value(expr) = value else {
            return Ok(None);
        };
        let declared = |local: &str| {
            rest.iter().any(|stmt| {
                matches!(
                    &stmt.kind,
                    StmtKind::RegisterDecl {
                        kind: ast::RegisterKind::Classical,
                        name,
                        ..
                    } if *name == local
                )
            })
        };
        match expr.as_ident() {
            Some(local) if is_bit_param(args, local) => Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "def `{name}` returning its bit parameter `{local}`, a copy the \
                     instruction list has no classical move for"
                ),
                line: last.line,
            }),
            Some(local) if declared(local) => Ok(Some(local)),
            _ => Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "return of `{expr}` from def `{name}`; a def returns a measurement or a \
                     bit it declares"
                ),
                line: last.line,
            }),
        }
    }

    /// Check that a `def` body writes classical bits only through its result,
    /// or under [`parse_dynamic`] also through the bits it declares, which
    /// `locals` collects as the walk meets them.
    ///
    /// A bit parameter is a copy, and outside [`parse_dynamic`] any other bit
    /// would need a classical register the program never declared, so neither
    /// can be written here.
    fn check_def_writes(
        name: &str,
        args: &[DefParam<'a>],
        local: Option<&str>,
        block: &Block<'a>,
        top: bool,
        mut locals: Option<Vec<&'a str>>,
    ) -> Result<Option<Vec<&'a str>>> {
        with_stack(|| {
            for stmt in block {
                match &stmt.kind {
                    StmtKind::RegisterDecl {
                        kind: ast::RegisterKind::Classical,
                        name: declared,
                        ..
                    } if !(top && local == Some(*declared)) && locals.is_some() => {
                        locals.as_mut().expect("checked").push(declared);
                    }
                    StmtKind::RegisterDecl {
                        kind: ast::RegisterKind::Classical,
                        name: declared,
                        ..
                    } if !(top && local == Some(*declared)) => {
                        return Err(PrismError::UnsupportedConstruct {
                            construct: format!(
                                "`bit {declared}` inside def `{name}`, which does not return it; \
                                 holding it would need a classical bit the program never declared"
                            ),
                            line: stmt.line,
                        });
                    }
                    StmtKind::Measure(measure) => {
                        let target = measure.target.register();
                        if target.is_some() && target == local {
                            continue;
                        }
                        if let (Some(target), Some(locals)) = (target, &locals)
                            && locals.contains(&target)
                            && !is_bit_param(args, target)
                        {
                            continue;
                        }
                        let construct = match target {
                            Some(param) if is_bit_param(args, param) => format!(
                                "measure into bit parameter `{param}` of def `{name}`, which is \
                                 passed by value"
                            ),
                            _ => format!(
                                "measure into `{}` inside def `{name}`, which writes classical \
                                 bits only through its result",
                                measure.target.describe()
                            ),
                        };
                        return Err(PrismError::UnsupportedConstruct {
                            construct,
                            line: stmt.line,
                        });
                    }
                    StmtKind::If(conditional) => {
                        locals = Self::check_def_writes(
                            name,
                            args,
                            local,
                            &conditional.then_body,
                            false,
                            locals,
                        )?;
                        if let Some(body) = &conditional.else_body {
                            locals =
                                Self::check_def_writes(name, args, local, body, false, locals)?;
                        }
                    }
                    StmtKind::For { body, .. }
                    | StmtKind::While { body, .. }
                    | StmtKind::Box { body, .. } => {
                        locals = Self::check_def_writes(name, args, local, body, false, locals)?;
                    }
                    StmtKind::Switch { arms, .. } => {
                        for arm in arms {
                            locals = Self::check_def_writes(
                                name, args, local, &arm.body, false, locals,
                            )?;
                        }
                    }
                    _ => {}
                }
            }
            Ok(locals)
        })
    }
}
