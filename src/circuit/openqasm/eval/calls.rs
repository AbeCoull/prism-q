//! Gate calls, subroutine calls and their expansion.

use super::*;

/// A `mutable` array argument's source array and the offsets its elements came from.
type ArrayWriteBack = (String, Vec<usize>);

/// What the arguments of one inlined `def` call bind, gathered before its body runs.
#[derive(Default)]
struct DefArguments<'a> {
    bindings: HashMap<&'a str, ClassicalValue>,
    symbolic: Vec<&'a str>,
    /// Arguments a runtime call passes as runtime values.
    scalar_args: Vec<(&'a str, ClassicalType, ClassicalExpr)>,
    durations: HashMap<&'a str, Timed>,
    complex: HashMap<&'a str, Complex64>,
    array_args: Vec<(&'a str, Rc<ClassicalArray>, Option<ArrayWriteBack>)>,
    qubit_bindings: Vec<(&'a str, SmallVec<[usize; 4]>)>,
    bit_bindings: Vec<(&'a str, SmallVec<[usize; 4]>)>,
}

impl<'a> Parser<'a> {
    pub(super) fn exec_call(
        &mut self,
        modifiers: &[ast::Modifier],
        name: &str,
        params: &[Argument<'a>],
        operands: &[Operand],
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let resolved = self.resolve_arguments(params, line, out)?;
        let params = resolved.as_ref();
        if self.externs.contains_key(name) {
            return self.exec_extern_statement(modifiers, name, params, operands, line);
        }
        if self.dynamic.is_some()
            && !self.def_defs.contains_key(name)
            && params.iter().any(|param| self.argument_is_runtime(param))
        {
            return self.exec_runtime_rotation(modifiers, name, params, operands, line, out);
        }
        let modifiers = self.fold_modifiers(modifiers, line)?;

        if self.def_defs.contains_key(name) {
            if !modifiers.is_empty() && self.def_is_runtime(name, &self.def_defs[name].args, params)
            {
                return Err(unsupported(
                    line,
                    format!("a modifier on runtime subroutine `{name}`"),
                ));
            }
            if self.streaming() {
                self.flush(std::mem::take(out));
            }
            let (instrs, _) = self.expand_def(name, params, None, line)?;
            if !modifiers.is_empty()
                && instrs
                    .iter()
                    .any(|instr| gate_angle(instr).and_then(tag_id).is_some())
            {
                return Err(unsupported(
                    line,
                    format!("modifier on `{name}` reading an `input`"),
                ));
            }
            out.extend(Self::modify_expansion(instrs, &modifiers, name, line)?);
            return Ok(());
        }

        let values = self.fold_params(params, line)?;
        let has_input = values.iter().any(|value| tag_id(*value).is_some());
        if has_input && (name == "gphase" || !modifiers.is_empty()) {
            let what = if name == "gphase" {
                "`gphase`".to_string()
            } else {
                format!("modifier on `{name}`")
            };
            return Err(unsupported(line, format!("{what} reading an `input`")));
        }
        if name == "gphase" {
            let qubits = operands
                .iter()
                .map(|operand| match self.qubits_of(operand)?.as_slice() {
                    [single] => Ok(*single),
                    _ => Err(parse_error(
                        line,
                        format!(
                            "`{}` names a register where one qubit belongs",
                            operand.describe()
                        ),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            out.extend(self.resolve_global_phase(&values, &modifiers, &qubits, line)?);
            return Ok(());
        }

        let resolved: SmallVec<[SmallVec<[usize; 4]>; 4]> = operands
            .iter()
            .map(|operand| self.qubits_of(operand))
            .collect::<Result<_>>()?;
        let width = self.broadcast_length(&resolved, name, line)?;

        for at in 0..width {
            let qubits: SmallVec<[usize; 4]> = resolved
                .iter()
                .map(|entry| {
                    if entry.len() == 1 {
                        entry[0]
                    } else {
                        entry[at]
                    }
                })
                .collect();
            self.resolve_gate_application_once(
                name, &values, &modifiers, &qubits, has_input, line, out,
            )?;
        }
        Ok(())
    }

    fn fold_modifiers(
        &self,
        modifiers: &[ast::Modifier],
        line: usize,
    ) -> Result<Vec<super::Modifier>> {
        let mut out = Vec::with_capacity(modifiers.len());
        let mut controls = 0usize;
        for modifier in modifiers {
            match modifier {
                ast::Modifier::Inv => out.push(super::Modifier::Inv),
                ast::Modifier::Ctrl { negated, count } => {
                    let count = match count {
                        None => 1,
                        Some(expr) => self.control_count(expr, line)?,
                    };
                    // Checked over the whole chain before any entry is built, so
                    // a long run of `ctrl(255) @` costs nothing to refuse.
                    controls += count;
                    if controls > usize::from(u8::MAX) {
                        return Err(unsupported(
                            line,
                            format!("ctrl @ chain past {} controls", u8::MAX),
                        ));
                    }
                    out.extend((0..count).map(|_| super::Modifier::Ctrl { negated: *negated }));
                }
                ast::Modifier::Pow(expr) => {
                    let exponent = self.value_of(expr, line)?;
                    if exponent.fract() == 0.0 && exponent.abs() > MAX_POW_REPEATS as f64 {
                        return Err(parse_error(
                            line,
                            format!(
                                "pow({exponent}) repeats a gate more than \
                                 {MAX_POW_REPEATS} times"
                            ),
                        ));
                    }
                    out.push(super::Modifier::Pow(exponent));
                }
            }
        }
        Ok(out)
    }

    /// The `n` of `ctrl(n) @`: a positive integer, at most the widest control
    /// chain a gate can carry.
    fn control_count(&self, expr: &Expr, line: usize) -> Result<usize> {
        let count = self.integer_of(expr, line)?;
        if count < 1 {
            return Err(parse_error(
                line,
                format!("`ctrl({expr})` needs a positive control count, got {count}"),
            ));
        }
        if count > i64::from(u8::MAX) {
            return Err(unsupported(
                line,
                format!("ctrl @ chain past {} controls", u8::MAX),
            ));
        }
        Ok(count as usize)
    }

    /// Initialize a fresh `bit` register from a bit string, written most
    /// significant first, or a value. The register starts cleared, so only an
    /// initializer that sets a bit writes one, which only a dynamic program can.
    pub(super) fn init_bits(
        &mut self,
        name: &str,
        size: Option<&Expr<'a>>,
        init: &ast::BitInit<'a>,
        line: usize,
    ) -> Result<()> {
        let width = match size {
            Some(size) => self.integer_of(size, line)?,
            None => 1,
        };
        let bits = self.bits_of(&Operand {
            name: OperandName::Register(name),
            index: None,
            line,
        })?;
        let values: Vec<bool> = match init {
            ast::BitInit::Bits(text) => {
                let digits: Vec<char> = text.chars().filter(|c| *c != '_').collect();
                if let Some(bad) = digits.iter().find(|c| !matches!(c, '0' | '1')) {
                    return Err(parse_error(
                        line,
                        format!("`\"{text}\"` is not a bit string: `{bad}` is neither 0 nor 1"),
                    ));
                }
                if digits.len() as i64 != width {
                    return Err(parse_error(
                        line,
                        format!(
                            "`\"{text}\"` has {} bit(s) but `{name}` declares {width}",
                            digits.len()
                        ),
                    ));
                }
                digits.iter().rev().map(|digit| *digit == '1').collect()
            }
            ast::BitInit::Value(expr) if self.reads_runtime(expr) => {
                let value = self.lower_expr(expr, line)?;
                return self.write_classical_bits(&bits, value, line);
            }
            ast::BitInit::Value(expr) => {
                let word = self.int_of(expr, line)?;
                (0..bits.len())
                    .map(|at| {
                        if at < 128 {
                            (word >> at) & 1 == 1
                        } else {
                            word < 0
                        }
                    })
                    .collect()
            }
        };
        if !values.contains(&true)
            && !self
                .def_result
                .as_ref()
                .is_some_and(|frame| frame.runtime && frame.local == Some(name))
        {
            return Ok(());
        }
        if self.dynamic.is_none() {
            return Err(unsupported(
                line,
                format!(
                    "`{name}` initialized with a set bit; the instruction list has no classical \
                     write, and `openqasm::parse_dynamic` carries one"
                ),
            ));
        }
        self.write_constant_bits(&bits, &values, line)
    }

    /// Evaluate a call's angle arguments. Two reading an `input` are rejected: a
    /// slot is written onto the single angle a bindable gate carries.
    fn fold_params(&self, params: &[Argument], line: usize) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(params.len());
        let mut tagged = false;
        for param in params {
            let element = self.element_argument(param);
            let expr = match (param, &element) {
                (_, Some(element)) => element,
                (Argument::Value(expr), None) => expr,
                (Argument::Operand(_), None) => {
                    return Err(parse_error(
                        line,
                        "a gate angle cannot be a qubit reference",
                    ));
                }
            };
            let value = self.fold_angle(expr, line)?;
            if tag_id(value).is_some() {
                if tagged {
                    return Err(unsupported(line, "two `input` parameters on one gate"));
                }
                tagged = true;
            }
            values.push(value);
        }
        Ok(values)
    }

    /// An argument written `a[i]` parses as a qubit or bit reference; read it
    /// as an array element when `a` names an array.
    fn element_argument<'e>(&self, argument: &Argument<'e>) -> Option<Expr<'e>> {
        let Argument::Operand(operand) = argument else {
            return None;
        };
        let array = operand.register().filter(|name| {
            self.arrays.contains_key(name)
                || matches!(
                    self.classical.get(name),
                    Some(ClassicalDecl {
                        ty: ClassicalType::Int(_),
                        ..
                    })
                )
        })?;
        let index = match &operand.index {
            Some(Index::Single(index)) => index.clone(),
            Some(Index::Range(range)) => Expr::Range(range.clone()),
            _ => return None,
        };
        Some(Expr::Element(Box::new(syntax_expr::Element {
            array,
            indices: vec![index],
        })))
    }

    /// One angle argument. One that reads an `input` comes back as a tag for its
    /// affine form, or as a number when that form is constant.
    fn fold_angle(&self, expr: &Expr, line: usize) -> Result<f64> {
        if self.symbol_in(expr).is_none() {
            return self.value_of(expr, line);
        }
        let form = self.affine_of(expr, line)?;
        Ok(match form.slot {
            Some(_) if form.scale != 0.0 => self.tag(form),
            _ => form.offset,
        })
    }

    /// `expr` as an affine form over the one slot its tagged names read.
    fn affine_of(&self, expr: &Expr, line: usize) -> Result<Linear> {
        if self.symbol_in(expr).is_none() {
            return Ok(Linear {
                slot: None,
                scale: 0.0,
                offset: self.value_of(expr, line)?,
            });
        }
        let scaled = |form: Linear, by: f64| Linear {
            scale: form.scale * by,
            offset: form.offset * by,
            ..form
        };
        with_stack(|| match expr {
            Expr::Ident(name) => Ok(self.symbol(name).expect("symbol_in found it")),
            Expr::Negate(inner) => Ok(scaled(self.affine_of(inner, line)?, -1.0)),
            Expr::Binary { op, left, right } => {
                let (left, right) = (self.affine_of(left, line)?, self.affine_of(right, line)?);
                let sign = match op {
                    BinaryOp::Add => 1.0,
                    BinaryOp::Sub => -1.0,
                    BinaryOp::Mul if left.slot.is_none() => {
                        return Ok(scaled(right, left.offset));
                    }
                    BinaryOp::Mul if right.slot.is_none() => {
                        return Ok(scaled(left, right.offset));
                    }
                    BinaryOp::Div if right.slot.is_none() && right.offset != 0.0 => {
                        return Ok(Linear {
                            scale: left.scale / right.offset,
                            offset: left.offset / right.offset,
                            ..left
                        });
                    }
                    _ => return Err(non_affine(expr, line)),
                };
                if left.slot.is_some() && right.slot.is_some() && left.slot != right.slot {
                    return Err(unsupported(
                        line,
                        format!("the angle `{expr}`, which reads two `input`s"),
                    ));
                }
                Ok(Linear {
                    slot: left.slot.or(right.slot),
                    scale: left.scale + sign * right.scale,
                    offset: left.offset + sign * right.offset,
                })
            }
            _ => Err(non_affine(expr, line)),
        })
    }

    /// Expand a user `gate` by binding its qubit names to the call's qubits and
    /// its parameters to the call's values, then running the body.
    pub(in crate::circuit::openqasm) fn expand_user_gate(
        &self,
        name: &str,
        call_params: &[f64],
        call_qubits: &SmallVec<[usize; 4]>,
        line: usize,
    ) -> Result<Option<Vec<Instruction>>> {
        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(parse_error(
                line,
                format!(
                    "gate expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive gate definition for `{name}`"
                ),
            ));
        }
        let Some(def) = self.gate_defs.get(name) else {
            return Ok(None);
        };
        if call_params.len() != def.params.len() {
            return Err(parse_error(
                line,
                format!(
                    "gate `{name}` expects {} parameters, got {}",
                    def.params.len(),
                    call_params.len()
                ),
            ));
        }
        if call_qubits.len() != def.qubits.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.qubits.len(),
                got: call_qubits.len(),
            });
        }

        // A gate body sees its own parameters and nothing else, which is what
        // the language scopes it to.
        let bindings: HashMap<&'a str, ClassicalValue> = def
            .params
            .iter()
            .copied()
            .zip(
                call_params
                    .iter()
                    .map(|&value| ClassicalValue::Float(value)),
            )
            .collect();
        let mut sub = self.expansion_parser(bindings);
        if call_params.iter().any(|value| tag_id(*value).is_some()) {
            sub.symbolic = def.params.clone();
        }
        if let Some(&first) = call_qubits.first() {
            sub.phase_qubit = first;
        }
        for (&qubit_name, qubit) in def.qubits.iter().zip(call_qubits) {
            sub.params.insert(
                qubit_name,
                Alias {
                    kind: ast::RegisterKind::Qubit,
                    indices: vec![*qubit],
                },
            );
        }
        Ok(Some(sub.execute(&def.body)?))
    }

    /// Evaluate an `extern` call written as a statement and drop its value. A
    /// call over runtime values is dropped unevaluated, since nothing reads it.
    fn exec_extern_statement<'e>(
        &self,
        modifiers: &[ast::Modifier],
        name: &'e str,
        params: &[Argument<'e>],
        operands: &[Operand],
        line: usize,
    ) -> Result<()> {
        if !modifiers.is_empty() || !operands.is_empty() {
            return Err(parse_error(
                line,
                format!("extern `{name}` is a function, not a gate, and takes no qubits"),
            ));
        }
        let call = Expr::Call(Box::new(syntax_expr::Call {
            name,
            args: params
                .iter()
                .map(|param| match param {
                    Argument::Value(value) => Ok(value.clone()),
                    Argument::Operand(operand) => operand_reference(operand),
                })
                .collect::<Result<_>>()?,
        }));
        if !self.reads_runtime(&call) {
            self.typed_of(&call, line)?;
        }
        Ok(())
    }

    /// Assign a subroutine result or a builtin value to the caller's target.
    pub(super) fn exec_call_assign(
        &mut self,
        assign: &ast::CallAssign<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let args = self.resolve_arguments(&assign.args, line, out)?;
        if self.def_defs.contains_key(assign.name) && self.streaming() {
            self.flush(std::mem::take(out));
        }
        if let Some(def) = self.def_defs.get(assign.name) {
            if def.result.as_ref().is_some_and(|result| result.ty != "bit") {
                let (instructions, value) = self.expand_def(assign.name, &args, None, line)?;
                out.extend(instructions);
                if self.unwinds_end() {
                    return Ok(());
                }
                if self.streaming() {
                    self.flush(std::mem::take(out));
                }
                let value = value.expect("typed result");
                return self.assign_call_result(&assign.target, value, line);
            }
            let bits = self.bits_of(&assign.target)?;
            let (instructions, _) = self.expand_def(assign.name, &args, Some(&bits), line)?;
            out.extend(instructions);
            return Ok(());
        }
        if self.externs.contains_key(assign.name) {
            let value = Expr::Call(Box::new(syntax_expr::Call {
                name: assign.name,
                args: args
                    .iter()
                    .map(|arg| match arg {
                        Argument::Value(value) => Ok(value.clone()),
                        Argument::Operand(operand) => operand_reference(operand),
                    })
                    .collect::<Result<_>>()?,
            }));
            return self.assign_call_result(&assign.target, value, line);
        }
        let Some(target) = assign
            .target
            .register()
            .filter(|_| assign.target.index.is_none())
        else {
            return Err(parse_error(
                line,
                format!(
                    "`{}` is not a name an assignment can write",
                    assign.target.describe()
                ),
            ));
        };
        let args = args
            .iter()
            .map(|arg| match arg {
                Argument::Value(expr) => Ok(expr.clone()),
                Argument::Operand(operand) => Err(parse_error(
                    line,
                    format!(
                        "`{}` is a qubit or bit where a value belongs",
                        operand.describe()
                    ),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let value = Expr::Call(Box::new(syntax_expr::Call {
            name: assign.name,
            args,
        }));
        self.assign_classical(target, None, &value, line)
    }

    /// Write a call's scalar result to a variable, an array element or bits.
    fn assign_call_result(
        &mut self,
        target: &Operand<'a>,
        value: Expr<'a>,
        line: usize,
    ) -> Result<()> {
        let name = target
            .register()
            .ok_or_else(|| parse_error(line, "a scalar result needs a classical target"))?;
        if self.is_bit_name(name) {
            return self.assign_bits(target, None, &value, line);
        }
        match &target.index {
            None => self.assign_classical(name, None, &value, line),
            Some(Index::Single(index)) => self.assign_element(
                &ast::ElementAssign {
                    array: name,
                    indices: vec![index.clone()],
                    op: None,
                    value,
                },
                line,
            ),
            _ => Err(parse_error(line, "a scalar result needs one array element")),
        }
    }

    /// Inline a `def` call, binding each argument by the kind its declaration
    /// gave it and the result, when there is one, to `result`.
    pub(super) fn expand_def(
        &mut self,
        name: &str,
        args: &[Argument],
        result: Option<&[usize]>,
        line: usize,
    ) -> Result<(Vec<Instruction>, Option<Expr<'a>>)> {
        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(parse_error(
                line,
                format!(
                    "def expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive call to `{name}`"
                ),
            ));
        }
        let definitions = Rc::clone(&self.def_defs);
        let def = definitions.get(name).expect("checked by the caller");
        if args.len() != def.args.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.args.len(),
                got: args.len(),
            });
        }

        let copied = 2 * def.constants.len() + def.arrays.len() + args.len();
        self.charge(copied, line)?;
        let runtime = self.def_is_runtime(name, &def.args, args);
        let mut arguments = self.bind_def_arguments(name, def, args, runtime, line)?;
        let frame = self.def_frame(name, def, result, &arguments.bit_bindings, line)?;
        let scalar_args = std::mem::take(&mut arguments.scalar_args);
        let (mut sub, write_backs) = self.def_subparser(def, arguments, frame);
        if runtime {
            let value = self.execute_runtime_def(name, def, sub, scalar_args, line)?;
            return Ok((Vec::new(), value));
        }
        for param in &def.args {
            if let DefParam::Value { name, ty, width } = param {
                let kind = sub.classical_type(ty, width.as_ref(), line)?;
                Rc::make_mut(&mut sub.cregs).remove(name);
                Rc::make_mut(&mut sub.aliases).remove(name);
                sub.params.remove(name);
                Rc::make_mut(&mut sub.classical).insert(
                    name,
                    ClassicalDecl {
                        ty: kind,
                        constant: false,
                        value: None,
                    },
                );
            }
        }
        let instructions = sub.execute(&def.body)?;
        self.write_back_arrays(&sub, write_backs);
        let frame = sub.def_result.expect("installed above");
        if sub.exit == Some(Exit::End) {
            self.exec_end(line)?;
            let value = frame.value.or_else(|| {
                def.result
                    .is_some()
                    .then_some(Expr::Number(ClassicalValue::Int(0)))
            });
            return Ok((instructions, value));
        }
        if def.local.is_some() {
            self.charge(frame.bits.len(), line)?;
            let mut initialized = frame.bits.iter().map(|&bit| (bit, false)).collect();
            if self.returned_bits_need_initialization(&instructions, &mut initialized, line)?
                || initialized.values().any(|initialized| !initialized)
            {
                return Err(unsupported(
                    line,
                    format!(
                        "an initializer on the bit returned by def `{name}`; use `parse_dynamic`"
                    ),
                ));
            }
        }
        if def.result.is_some() && !frame.returned {
            return Err(parse_error(
                line,
                format!("def `{name}` reaches its end without a result"),
            ));
        }
        Ok((instructions, frame.value))
    }

    /// Bind each argument of a call to def `name`, in declaration order, with the
    /// def's constants seeded first.
    fn bind_def_arguments(
        &self,
        name: &str,
        def: &DefDefinition<'a>,
        args: &[Argument],
        runtime: bool,
        line: usize,
    ) -> Result<DefArguments<'a>> {
        let mut bound = DefArguments::default();
        for (&name, decl) in def.constants.iter() {
            match decl.value {
                Some(ConstValue::Number(value)) => {
                    bound.bindings.insert(name, value);
                }
                Some(ConstValue::Complex(value)) => {
                    bound.complex.insert(name, value);
                }
                Some(ConstValue::Duration(value)) => {
                    bound.durations.insert(name, value);
                }
                None => {}
            }
        }
        let mut bound_qubits = HashSet::new();
        for (slot, argument) in def.args.iter().zip(args) {
            match slot {
                DefParam::Qubit {
                    name: param_name,
                    width,
                } => self.bind_qubit_param(
                    name,
                    param_name,
                    width.as_ref(),
                    argument,
                    &mut bound_qubits,
                    &mut bound,
                    line,
                )?,
                DefParam::Value {
                    name: param_name,
                    ty,
                    width,
                } => self.bind_value_param(
                    name,
                    param_name,
                    ty,
                    width.as_ref(),
                    argument,
                    runtime,
                    &mut bound,
                    line,
                )?,
                DefParam::Duration(param_name) => {
                    self.bind_duration_param(name, param_name, argument, &mut bound, line)?
                }
                DefParam::Bit {
                    name: param_name,
                    width,
                } => self.bind_bit_param(
                    name,
                    param_name,
                    width.as_ref(),
                    argument,
                    runtime,
                    &mut bound,
                    line,
                )?,
                DefParam::Array {
                    name: param_name,
                    mutable,
                    ty,
                    width,
                    shape,
                } => self.bind_array_param(
                    name,
                    param_name,
                    *mutable,
                    ty,
                    width.as_ref(),
                    shape,
                    argument,
                    runtime,
                    &mut bound,
                    line,
                )?,
            }
        }
        Ok(bound)
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_qubit_param(
        &self,
        name: &str,
        param_name: &'a str,
        width: Option<&Expr>,
        argument: &Argument,
        bound_qubits: &mut HashSet<usize>,
        bound: &mut DefArguments<'a>,
        line: usize,
    ) -> Result<()> {
        let operand = match argument {
            Argument::Operand(operand) => operand.clone(),
            Argument::Value(expr) => {
                let Some(register) = expr.as_ident() else {
                    return Err(parse_error(
                        line,
                        format!(
                            "def `{name}` qubit parameter `{param_name}` needs a \
                             qubit, not an expression"
                        ),
                    ));
                };
                Operand {
                    name: OperandName::Register(register),
                    index: None,
                    line,
                }
            }
        };
        let qubits = self.qubits_of(&operand)?;
        self.charge(qubits.len(), line)?;
        let wanted = match width {
            Some(width) => self.integer_of(width, line)?,
            None => 1,
        };
        if wanted <= 0 || qubits.len() as i64 != wanted {
            return Err(parse_error(
                line,
                format!(
                    "def `{name}` qubit parameter `{param_name}` takes {wanted} qubit(s), got {}",
                    qubits.len()
                ),
            ));
        }
        if qubits.iter().any(|&qubit| !bound_qubits.insert(qubit)) {
            return Err(parse_error(
                line,
                format!("def `{name}` passes a qubit more than once"),
            ));
        }
        bound.qubit_bindings.push((param_name, qubits));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_value_param(
        &self,
        name: &str,
        param_name: &'a str,
        ty: &str,
        width: Option<&Expr>,
        argument: &Argument,
        runtime: bool,
        bound: &mut DefArguments<'a>,
        line: usize,
    ) -> Result<()> {
        let element = self.element_argument(argument);
        let expr = match (argument, &element) {
            (_, Some(element)) => element,
            (Argument::Value(expr), None) => expr,
            (Argument::Operand(_), None) => {
                return Err(parse_error(
                    line,
                    format!("def `{name}` parameter `{param_name}` takes a value"),
                ));
            }
        };
        let kind = self.classical_type(ty, width, line)?;
        // An `input` passes through as its tag, which the body reads
        // only as a gate angle. An angle parameter keeps it unwrapped,
        // as a bound `input angle` is.
        if let Some(input) = self.symbol_in(expr) {
            if !matches!(
                kind,
                ClassicalType::Float(64 | 128) | ClassicalType::Angle(None)
            ) {
                return Err(self.symbol_error(
                    input,
                    &format!("as `def {name}` parameter `{param_name}`, whose type rounds it"),
                    line,
                ));
            }
            bound.bindings.insert(
                param_name,
                ClassicalValue::Float(self.fold_angle(expr, line)?),
            );
            bound.symbolic.push(param_name);
            return Ok(());
        }
        if let ClassicalType::Complex(_) = kind {
            if runtime {
                return Err(super::dynamic::runtime_complex(line));
            }
            bound
                .complex
                .insert(param_name, self.complex_of(expr, line)?);
            return Ok(());
        }
        if runtime {
            let lowered = match self.lower_expr(expr, line)? {
                ClassicalExpr::Const(_) => ClassicalExpr::Const(self.fold_typed(kind, expr, line)?),
                value => value,
            };
            bound.scalar_args.push((param_name, kind, lowered));
        } else {
            bound
                .bindings
                .insert(param_name, self.fold_typed(kind, expr, line)?);
        }
        Ok(())
    }

    fn bind_duration_param(
        &self,
        name: &str,
        param_name: &'a str,
        argument: &Argument,
        bound: &mut DefArguments<'a>,
        line: usize,
    ) -> Result<()> {
        let Argument::Value(expr) = argument else {
            return Err(parse_error(
                line,
                format!("def `{name}` parameter `{param_name}` takes a duration"),
            ));
        };
        let expr = self.with_durationof(expr, line)?;
        let value = timed_of(
            &expr,
            line,
            self.param_vars.as_ref(),
            &self.durations,
            &self.arrays,
        )?;
        if let Timed::Number(number) = value {
            return Err(parse_error(
                line,
                format!("def `{name}` parameter `{param_name}` takes a duration, got {number}"),
            ));
        }
        bound.durations.insert(param_name, value);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_bit_param(
        &self,
        name: &str,
        param_name: &'a str,
        width: Option<&Expr>,
        argument: &Argument,
        runtime: bool,
        bound: &mut DefArguments<'a>,
        line: usize,
    ) -> Result<()> {
        if runtime && let Some((size, value)) = self.runtime_bit_argument(argument, line)? {
            let wanted = width
                .map(|width| self.integer_of(width, line))
                .transpose()?
                .unwrap_or(1);
            if size as i64 != wanted {
                return Err(parse_error(
                    line,
                    format!(
                        "def `{name}` bit parameter `{param_name}` takes {wanted} bit(s), got {size}"
                    ),
                ));
            }
            bound.scalar_args.push((
                param_name,
                ClassicalType::Int(RuntimeType::Uint { width: size as u32 }),
                value,
            ));
            return Ok(());
        }
        let operand = match argument {
            Argument::Operand(operand) => operand.clone(),
            Argument::Value(expr) => match expr.as_ident() {
                Some(register) => Operand {
                    name: OperandName::Register(register),
                    index: None,
                    line,
                },
                None => {
                    return Err(parse_error(
                        line,
                        format!(
                            "def `{name}` bit parameter `{param_name}` needs a bit, \
                             not an expression"
                        ),
                    ));
                }
            },
        };
        let bits = self.bits_of(&operand)?;
        let wanted = match width {
            Some(width) => self.integer_of(width, line)?,
            None => 1,
        };
        if bits.len() as i64 != wanted {
            return Err(parse_error(
                line,
                format!(
                    "def `{name}` bit parameter `{param_name}` takes {wanted} bit(s), \
                     got {}",
                    bits.len()
                ),
            ));
        }
        if runtime {
            let value = super::dynamic::bits_value(&bits, line)?;
            let kind = RuntimeType::Uint {
                width: bits.len() as u32,
            };
            bound
                .scalar_args
                .push((param_name, ClassicalType::Int(kind), value));
        } else {
            bound.bit_bindings.push((param_name, bits));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_array_param(
        &self,
        name: &str,
        param_name: &'a str,
        mutable: bool,
        ty: &str,
        width: Option<&Expr>,
        shape: &ArrayShape,
        argument: &Argument,
        runtime: bool,
        bound: &mut DefArguments<'a>,
        line: usize,
    ) -> Result<()> {
        let reference = match argument {
            Argument::Value(expr) => expr.clone(),
            Argument::Operand(operand) => operand_reference(operand)?,
        };
        let (source, indices) = match &reference {
            Expr::Ident(source) => (*source, &[][..]),
            Expr::Element(element) => (element.array, element.indices.as_slice()),
            _ => ("", &[][..]),
        };
        let Some(array) = self.arrays.get(source) else {
            return Err(parse_error(
                line,
                format!(
                    "def `{name}` parameter `{param_name}` takes an array, got \
                     `{reference}`"
                ),
            ));
        };
        let kind = self.array_kind(ty, width, line)?;
        if kind != array.ty {
            return Err(parse_error(
                line,
                format!(
                    "def `{name}` parameter `{param_name}` takes `{ty}` elements, \
                     which `{source}` does not hold"
                ),
            ));
        }
        let view = array_view(array, source, indices, line, |index| {
            self.typed_of(index, line)
        })?;
        let fits = match shape {
            ArrayShape::Sized(dims) => {
                let wanted = dims
                    .iter()
                    .map(|size| self.integer_of(size, line))
                    .collect::<Result<Vec<_>>>()?;
                wanted.len() == view.dims.len()
                    && wanted
                        .iter()
                        .zip(&view.dims)
                        .all(|(&wanted, &size)| wanted == size as i64)
            }
            ArrayShape::Rank(rank) => self.integer_of(rank, line)? == view.dims.len() as i64,
        };
        if !fits {
            return Err(parse_error(
                line,
                format!(
                    "def `{name}` parameter `{param_name}` cannot take `{reference}`, \
                     whose shape is {:?}",
                    view.dims
                ),
            ));
        }
        if mutable && array.constant {
            return Err(parse_error(
                line,
                format!("`{source}` is `const` and cannot pass as a `mutable` array"),
            ));
        }
        if mutable && runtime {
            return Err(unsupported(
                line,
                format!(
                    "a `mutable` array argument to def `{name}`, whose call runs \
                     at runtime"
                ),
            ));
        }
        self.charge(view.offsets.len(), line)?;
        let array = ClassicalArray {
            ty: kind,
            constant: !mutable,
            dims: view.dims,
            values: view.offsets.iter().map(|&at| array.values[at]).collect(),
        };
        let write_back = mutable.then(|| (source.to_string(), view.offsets));
        bound
            .array_args
            .push((param_name, Rc::new(array), write_back));
        Ok(())
    }

    /// The result frame a call to def `name` runs under, with `result` naming the
    /// bits it assigns.
    fn def_frame(
        &self,
        name: &str,
        def: &DefDefinition<'a>,
        result: Option<&[usize]>,
        bit_bindings: &[(&'a str, SmallVec<[usize; 4]>)],
        line: usize,
    ) -> Result<DefResult<'a>> {
        let dropped: SmallVec<[usize; 4]>;
        let result = match (&def.result, result) {
            (Some(declared), None) if declared.ty == "bit" && self.private_bits.is_some() => {
                let width = match &declared.width {
                    Some(width) => self.integer_of(width, line)?.max(1) as usize,
                    None => 1,
                };
                let offset = self.private_register(width, line)?;
                dropped = (offset..offset + width).collect();
                Some(dropped.as_slice())
            }
            _ => result,
        };
        Ok(match (&def.result, result) {
            (None, None) => DefResult {
                bits: SmallVec::new(),
                local: None,
                ty: None,
                value: None,
                returned: false,
                runtime: false,
                result_var: None,
            },
            (None, Some(_)) => {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares no result to assign"),
                ));
            }
            (Some(declared), None) if declared.ty != "bit" => DefResult {
                bits: SmallVec::new(),
                local: None,
                ty: Some(self.classical_type(declared.ty, declared.width.as_ref(), line)?),
                value: None,
                returned: false,
                runtime: false,
                result_var: None,
            },
            (Some(_), None) => {
                return Err(unsupported(
                    line,
                    format!(
                        "a call to def `{name}` that drops its `bit` result, which would need \
                         a classical bit the program never declared"
                    ),
                ));
            }
            (Some(declared), Some(bits)) => {
                let wanted = match &declared.width {
                    Some(width) => self.integer_of(width, line)?,
                    None => 1,
                };
                if bits.len() as i64 != wanted {
                    return Err(parse_error(
                        line,
                        format!(
                            "def `{name}` returns {wanted} bit(s) where its target names {}",
                            bits.len()
                        ),
                    ));
                }
                // The named result is written in place, so a parameter reading
                // the same bit would see the new value rather than its copy.
                let aliased = def.local.is_some()
                    && bit_bindings
                        .iter()
                        .any(|(_, param)| param.iter().any(|bit| bits.contains(bit)));
                if aliased {
                    return Err(unsupported(
                        line,
                        format!(
                            "a call to def `{name}` passing a bit it also assigns, which the \
                             body would read after writing its result"
                        ),
                    ));
                }
                DefResult {
                    bits: bits.iter().copied().collect(),
                    local: def.local,
                    ty: None,
                    value: None,
                    returned: false,
                    runtime: false,
                    result_var: None,
                }
            }
        })
    }

    /// The parser a def body runs in, holding its bound arguments and `frame`,
    /// and the `mutable` arrays to copy back once the body has run.
    fn def_subparser(
        &self,
        def: &DefDefinition<'a>,
        arguments: DefArguments<'a>,
        frame: DefResult<'a>,
    ) -> (Parser<'a>, Vec<(&'a str, ArrayWriteBack)>) {
        let mut sub = self.expansion_parser(arguments.bindings);
        sub.symbolic = arguments.symbolic;
        // The body runs in a parser of its own, so a write it made to an outer
        // variable would be dropped on return; only the constants it may read
        // stay declared, and a write to anything else is rejected.
        sub.classical = Rc::new(
            def.constants
                .iter()
                .filter(|(_, decl)| decl.constant)
                .map(|(name, decl)| (*name, decl.clone()))
                .collect(),
        );
        sub.durations = arguments.durations;
        sub.complex = arguments.complex;
        // A def sees the global constants and nothing else it does not take as
        // an argument, so only a `const` array reaches its body.
        sub.arrays = def.arrays.clone();
        let mut write_backs = Vec::new();
        for (param_name, array, write_back) in arguments.array_args {
            sub.arrays.insert(param_name, array);
            if let Some(write_back) = write_back {
                write_backs.push((param_name, write_back));
            }
        }
        sub.qregs = Rc::new(HashMap::new());
        sub.cregs = Rc::new(HashMap::new());
        sub.aliases = Rc::new(HashMap::new());
        for (param_name, qubits) in arguments.qubit_bindings {
            sub.params.insert(
                param_name,
                Alias {
                    kind: ast::RegisterKind::Qubit,
                    indices: qubits.into_vec(),
                },
            );
        }
        for (param_name, bits) in arguments.bit_bindings {
            sub.bind_bits(param_name, bits);
        }
        if let Some(local) = frame.local {
            sub.bind_bits(local, frame.bits.clone());
        }
        sub.def_result = Some(frame);
        (sub, write_backs)
    }

    fn write_back_arrays(&mut self, sub: &Parser<'a>, write_backs: Vec<(&'a str, ArrayWriteBack)>) {
        for (param_name, (source, offsets)) in write_backs {
            if let (Some(written), Some(array)) = (
                sub.arrays.get(param_name),
                self.arrays.get_mut(source.as_str()),
            ) {
                let array = Rc::make_mut(array);
                for (&at, &value) in offsets.iter().zip(&written.values) {
                    array.values[at] = value;
                }
            }
        }
    }

    fn returned_bits_need_initialization(
        &self,
        body: &[Instruction],
        initialized: &mut HashMap<usize, bool>,
        line: usize,
    ) -> Result<bool> {
        with_stack(|| {
            let mut read = Vec::new();
            for instruction in body {
                let condition = match instruction {
                    Instruction::Measure { classical_bit, .. } => {
                        if let Some(bit) = initialized.get_mut(classical_bit) {
                            *bit = true;
                        }
                        continue;
                    }
                    Instruction::Conditional { condition, .. } => Some(condition),
                    Instruction::Region(region) => Some(region.condition()),
                    _ => None,
                };
                if let Some(condition) = condition {
                    read.clear();
                    crate::circuit::collect_condition_bits(condition, &mut read);
                    self.charge(read.len(), line)?;
                    if read.iter().any(|bit| initialized.get(bit) == Some(&false)) {
                        return Ok(true);
                    }
                }
                if let Instruction::Region(region) = instruction {
                    self.charge(initialized.len(), line)?;
                    if self.returned_bits_need_initialization(
                        region.body(),
                        &mut initialized.clone(),
                        line,
                    )? {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        })
    }

    /// Name `bits` inside an expanded body: as a register when they run
    /// contiguously, which a register comparison needs, and as an alias
    /// otherwise.
    fn bind_bits(&mut self, name: &'a str, bits: SmallVec<[usize; 4]>) {
        let contiguous = bits.windows(2).all(|pair| pair[1] == pair[0] + 1);
        match bits.first() {
            Some(&offset) if contiguous => {
                self.params.remove(name);
                if self.aliases.contains_key(name) {
                    Rc::make_mut(&mut self.aliases).remove(name);
                }
                Rc::make_mut(&mut self.cregs).insert(
                    name,
                    Register {
                        offset,
                        size: bits.len(),
                    },
                );
            }
            _ => {
                self.params.insert(
                    name,
                    Alias {
                        kind: ast::RegisterKind::Classical,
                        indices: bits.to_vec(),
                    },
                );
            }
        }
    }

    /// A parser for an expanded body: the enclosing registers and definitions,
    /// one level deeper, with its own bindings.
    fn expansion_parser(&self, bindings: HashMap<&'a str, ClassicalValue>) -> Parser<'a> {
        Parser {
            dialect: self.dialect,
            input: "",
            qregs: Rc::clone(&self.qregs),
            cregs: Rc::clone(&self.cregs),
            gate_defs: Rc::clone(&self.gate_defs),
            def_defs: Rc::clone(&self.def_defs),
            externs: Rc::clone(&self.externs),
            registry: self.registry.clone(),
            total_qubits: self.total_qubits,
            total_cbits: self.total_cbits,
            gate_expansion_depth: self.gate_expansion_depth + 1,
            region_depth: self.region_depth,
            exec_depth: self.exec_depth,
            param_vars: Some(bindings),
            inputs: HashMap::new(),
            input_names: Vec::new(),
            tags: Rc::clone(&self.tags),
            symbolic: Vec::new(),
            supplied: Vec::new(),
            nested: true,
            results: Vec::new(),
            noise_specs: Vec::new(),
            pending_noise: None,
            verbatim_pending: false,
            physical: self.physical,
            aliases: Rc::clone(&self.aliases),
            params: HashMap::new(),
            classical: Rc::clone(&self.classical),
            durations: HashMap::new(),
            complex: HashMap::new(),
            arrays: HashMap::new(),
            declared: Vec::new(),
            def_result: None,
            loops: Vec::new(),
            exit: None,
            guard: None,
            dynamic: None,
            expansion_left: Rc::clone(&self.expansion_left),
            private_bits: self.private_bits.clone(),
            in_outer_while: self.in_outer_while || self.in_while(),
            phase_qubit: self.phase_qubit,
            stretch_ids: Rc::clone(&self.stretch_ids),
        }
    }
}
