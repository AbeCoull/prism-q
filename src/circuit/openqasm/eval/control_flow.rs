//! Branches, loops, `box` and scoped blocks.

use super::*;

impl<'a> Parser<'a> {
    pub(super) fn exec_if(
        &mut self,
        conditional: &ast::Conditional<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<Vec<Instruction>> {
        if let Ok(expr) = super::dynamic::condition_expr(&conditional.condition)
            && expr
                .finds_call(&|name| self.def_defs.contains_key(name))
                .is_some()
        {
            let value = self.resolve_calls(&expr, line, out)?;
            let ClassicalExpr::Const(value) = self.lower_expr(&value, line)? else {
                return Err(unsupported(
                    line,
                    "a def call in a measurement-dependent condition",
                ));
            };
            let body = if value.truthy() {
                Some(&conditional.then_body)
            } else {
                conditional.else_body.as_ref()
            };
            return body.map_or(Ok(Vec::new()), |body| self.exec_box(body));
        }

        let general = match &conditional.condition {
            Condition::Expr(_) => true,
            Condition::Truthy(operand)
            | Condition::Negated(operand)
            | Condition::Compare { lhs: operand, .. } => {
                self.single_bit(operand).is_none()
                    && operand.register().is_some_and(|name| {
                        !self.is_bit_name(name)
                            && (self.classical.contains_key(name)
                                || self.arrays.contains_key(name)
                                || syntax_expr::eval(&Expr::Ident(name), line, None).is_ok()
                                || self
                                    .param_vars
                                    .as_ref()
                                    .is_some_and(|vars| vars.contains_key(name)))
                    })
            }
            Condition::Parity { .. } => false,
        };
        if general {
            let expr = super::dynamic::condition_expr(&conditional.condition)?;
            if let ClassicalExpr::Const(value) = self.lower_expr(&expr, line)? {
                let body = if value.truthy() {
                    Some(&conditional.then_body)
                } else {
                    conditional.else_body.as_ref()
                };
                return body.map_or(Ok(Vec::new()), |body| self.exec_box(body));
            }
        }
        let condition = self.condition_of(&conditional.condition, line)?;
        let then_instrs = self.region(&conditional.then_body)?;
        let Some(else_body) = &conditional.else_body else {
            return Ok(guarded(condition, then_instrs).into_iter().collect());
        };
        // The negated region re-reads the classical bits after the `then` body
        // has run, so a body that measures into its own guard bits could take
        // both arms.
        if crate::circuit::body_writes_condition_bits(&then_instrs, &condition) {
            return Err(parse_error(
                line,
                "`else` needs a condition the `if` body does not overwrite; \
                          this body measures into a bit the condition reads",
            ));
        }
        let else_instrs = self.region(else_body)?;
        let mut out = Vec::new();
        out.extend(guarded(condition.clone(), then_instrs));
        out.extend(guarded(condition.negate(), else_instrs));
        Ok(out)
    }

    /// Run a `box` body in place, with no guard and no barrier at either edge.
    ///
    /// A box only fixes timing, which an ideal simulation does not model, so
    /// fusion may merge across its boundary as freely as across a `delay`. The
    /// names its body declares go out of scope with it.
    pub(in crate::circuit::openqasm) fn exec_box(
        &mut self,
        body: &Block<'a>,
    ) -> Result<Vec<Instruction>> {
        let scope = self.open_scope();
        let was_nested = std::mem::replace(&mut self.nested, true);
        let result = self.execute(body);
        self.nested = was_nested;
        self.close_scope(scope);
        result
    }

    /// The point to hand [`Parser::close_scope`] when the scope ends.
    pub(in crate::circuit::openqasm) fn open_scope(&self) -> usize {
        self.declared.len()
    }

    /// Drop the names declared since `scope` was opened. Names declared before
    /// it keep whatever the body wrote to them.
    pub(in crate::circuit::openqasm) fn close_scope(&mut self, scope: usize) {
        for name in self.declared.split_off(scope) {
            self.forget_runtime(name);
            Rc::make_mut(&mut self.classical).remove(name);
            self.durations.remove(name);
            self.complex.remove(name);
            self.arrays.remove(name);
            if let Some(values) = self.param_vars.as_mut() {
                values.remove(name);
            }
        }
    }

    /// Run a guarded block one nesting level down, in a scope of its own.
    ///
    /// Whether the block runs is decided by a measurement, while a classical
    /// variable is folded here at parse time, so the block may not write any
    /// variable declared outside it.
    pub(in crate::circuit::openqasm) fn region(
        &mut self,
        block: &Block<'a>,
    ) -> Result<Vec<Instruction>> {
        self.enter_region_depth(block)?;
        let scope = self.open_scope();
        let outer_guard = self.guard.replace(scope);
        let was_nested = std::mem::replace(&mut self.nested, true);
        self.enter_collecting();
        let body = self.execute(block).and_then(|body| {
            if body
                .iter()
                .any(|instr| gate_angle(instr).and_then(tag_id).is_some())
            {
                return Err(unsupported(
                    block.first().map_or(1, |stmt| stmt.line),
                    "an angle reading an `input` inside a guarded region",
                ));
            }
            Ok(body)
        });
        self.leave_collecting();
        self.nested = was_nested;
        self.guard = outer_guard;
        self.close_scope(scope);
        self.region_depth -= 1;
        body
    }

    pub(super) fn reject_guarded_write(&self, name: &str, line: usize) -> Result<()> {
        let Some(guard) = self.guard else {
            return Ok(());
        };
        let outer = self.classical.contains_key(name) || self.arrays.contains_key(name);
        if outer && !self.declared[guard..].contains(&name) {
            return Err(unsupported(
                line,
                format!(
                    "a write to `{name}` under a runtime `if` or `switch`; the variable is \
                     folded at parse time and cannot depend on a measurement, so parse the \
                     program with `openqasm::parse_dynamic` to evaluate it at runtime"
                ),
            ));
        }
        Ok(())
    }

    fn enter_region_depth(&mut self, block: &Block<'a>) -> Result<()> {
        if self.region_depth >= MAX_REGION_DEPTH {
            return Err(parse_error(
                block.first().map_or(1, |stmt| stmt.line),
                format!("`if` blocks nest deeper than {MAX_REGION_DEPTH}"),
            ));
        }
        self.region_depth += 1;
        Ok(())
    }

    pub(super) fn exec_for(
        &mut self,
        stmt: &Stmt<'a>,
        variable: &'a str,
        range: &ForRange<'a>,
        body: &Block<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let range = self.resolve_range_calls(range, line, out)?;
        let values = self.for_values(&range, line)?;
        if values.len() as i64 > MAX_FOR_ITERATIONS {
            return Err(parse_error(
                line,
                format!(
                    "for loop iterates {} times (max {MAX_FOR_ITERATIONS})",
                    values.len()
                ),
            ));
        }
        // The loop variable may shadow an outer name, whose value comes back
        // once the loop ends.
        let shadowed = self
            .param_vars
            .as_ref()
            .and_then(|vars| vars.get(variable).copied());
        let runtime_binding = self
            .dynamic
            .as_mut()
            .and_then(|state| state.vars.remove(variable));
        let bit_binding = self
            .dynamic
            .as_mut()
            .and_then(|state| state.bit_values.remove(variable));
        let shadowed_decl = Rc::make_mut(&mut self.classical).insert(
            variable,
            ClassicalDecl {
                ty: ClassicalType::Int(RuntimeType::Int { width: 64 }),
                constant: false,
                value: None,
            },
        );
        let scope = self.open_scope();
        let mut result = Ok(());
        self.enter_static_loop();
        let graph = self.unrolls_into_graph(stmt);
        if graph {
            self.flush(std::mem::take(out));
            self.begin_unrolled();
        }
        self.loops.push(Loop::For {
            region_depth: self.region_depth,
            frames: self.graph_depth(),
            graph,
        });
        for (pass, value) in values.into_iter().enumerate() {
            if let Err(error) = self.charge(1, line) {
                result = Err(error);
                break;
            }
            let mark = if graph { self.pass_mark() } else { (0, 0) };
            if graph
                && pass > 0
                && let Err(error) = self.next_pass()
            {
                result = Err(error);
                break;
            }
            // Each pass drops what the body declared, so a declaration binds
            // for one pass, while a write to an outer name carries on.
            self.param_vars
                .get_or_insert_with(HashMap::new)
                .insert(variable, ClassicalValue::Int(i128::from(value)));
            let was_nested = std::mem::replace(&mut self.nested, true);
            let produced = self.execute(body);
            self.nested = was_nested;
            self.close_scope(scope);
            match produced {
                Ok(produced) => {
                    if graph {
                        self.flush(produced);
                        if let Err(error) = self.charge_pass_blocks(mark, line) {
                            result = Err(error);
                            break;
                        }
                    } else {
                        out.extend(produced);
                    }
                    // A path a runtime `continue` sent to the next pass still
                    // needs that pass, so a parse-time exit after one jumps
                    // instead of ending the unrolling.
                    let continued = graph && self.pass_continued();
                    match self.exit {
                        Some(Exit::Break) => {
                            self.exit = None;
                            if !continued {
                                break;
                            }
                            if let Err(error) = self.break_unrolled() {
                                result = Err(error);
                                break;
                            }
                        }
                        Some(Exit::Continue) => self.exit = None,
                        Some(Exit::End) => break,
                        None => {}
                    }
                    if let Some(frame) = self.def_result.as_mut().filter(|frame| frame.returned) {
                        if !continued {
                            break;
                        }
                        frame.returned = false;
                    }
                }
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.loops.pop();
        if graph && result.is_ok() {
            // Paths a runtime `break` sent to the exit go on past the loop,
            // whatever `return` the rest took.
            if self.loop_broke()
                && let Some(frame) = self.def_result.as_mut()
            {
                frame.returned = false;
            }
            result = self.end_unrolled();
        }
        self.leave_static_loop();
        if let Some(state) = self.dynamic.as_mut() {
            if let Some(var) = runtime_binding {
                state.vars.insert(variable, var);
            }
            if let Some(width) = bit_binding {
                state.bit_values.insert(variable, width);
            }
        }
        match shadowed_decl {
            Some(decl) => {
                Rc::make_mut(&mut self.classical).insert(variable, decl);
            }
            None => {
                Rc::make_mut(&mut self.classical).remove(variable);
            }
        }
        if let Some(vars) = self.param_vars.as_mut() {
            match shadowed {
                Some(value) => vars.insert(variable, value),
                None => vars.remove(variable),
            };
        }
        result
    }

    /// `range` with each `def` call in its bounds expanded ahead of the loop, in
    /// source order.
    fn resolve_range_calls<'r>(
        &mut self,
        range: &'r ForRange<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<Cow<'r, ForRange<'a>>> {
        let calls_def = |expr: &Expr| {
            expr.finds_call(&|name| self.def_defs.contains_key(name))
                .is_some()
        };
        let any_call = match range {
            ForRange::Set(entries) => entries.iter().any(calls_def),
            ForRange::Range { start, step, stop } => {
                calls_def(start) || step.as_ref().is_some_and(calls_def) || calls_def(stop)
            }
        };
        if !any_call {
            return Ok(Cow::Borrowed(range));
        }
        Ok(Cow::Owned(match range {
            ForRange::Set(entries) => ForRange::Set(
                entries
                    .iter()
                    .map(|entry| self.resolve_calls(entry, line, out))
                    .collect::<Result<_>>()?,
            ),
            ForRange::Range { start, step, stop } => ForRange::Range {
                start: self.resolve_calls(start, line, out)?,
                step: step
                    .as_ref()
                    .map(|step| self.resolve_calls(step, line, out))
                    .transpose()?,
                stop: self.resolve_calls(stop, line, out)?,
            },
        }))
    }

    fn for_values(&self, range: &ForRange, line: usize) -> Result<Vec<i64>> {
        match range {
            ForRange::Set(entries) => entries
                .iter()
                .map(|entry| self.integer_of(entry, line))
                .collect(),
            ForRange::Range { start, step, stop } => {
                let first = self.integer_of(start, line)?;
                let last = self.integer_of(stop, line)?;
                let stride = match step {
                    None => 1,
                    Some(expr) => self.integer_of(expr, line)?,
                };
                if stride == 0 {
                    return Err(parse_error(line, "for loop range step must be non-zero"));
                }
                let mut values = Vec::new();
                let mut at = first;
                while (stride > 0 && at <= last) || (stride < 0 && at >= last) {
                    values.push(at);
                    if values.len() as i64 > MAX_FOR_ITERATIONS {
                        return Err(parse_error(
                            line,
                            format!("for loop iterates more than {MAX_FOR_ITERATIONS} times"),
                        ));
                    }
                    let Some(next) = at.checked_add(stride) else {
                        break;
                    };
                    at = next;
                }
                Ok(values)
            }
        }
    }

    pub(super) fn exec_switch(
        &mut self,
        target: &Expr<'a>,
        arms: &[ast::SwitchArm<'a>],
        line: usize,
    ) -> Result<Vec<Instruction>> {
        let lowered = self.lower_expr(target, line)?;
        if let ClassicalExpr::Const(value) = lowered {
            return self.exec_switch_constant(value, arms, line);
        }
        let (offset, size) = register_value(&lowered).ok_or_else(|| {
            unsupported(
                line,
                format!(
                    "a `switch` on `{target}`, which no guard on classical bits spells; \
                     `openqasm::parse_dynamic` evaluates it at runtime"
                ),
            )
        })?;
        let mut labels: Vec<u64> = Vec::new();
        let mut cases: Vec<(Vec<u64>, Vec<Instruction>)> = Vec::new();
        let mut default: Option<Vec<Instruction>> = None;
        for arm in arms {
            let body = self.region(&arm.body)?;
            let Some(arm_labels) = &arm.labels else {
                if default.replace(body).is_some() {
                    return Err(parse_error(
                        arm.line,
                        "`switch` has more than one `default` arm",
                    ));
                }
                continue;
            };
            let mut values = Vec::new();
            for label in arm_labels {
                let value = self.integer_of(label, arm.line)?;
                let value = u64::try_from(value).map_err(|_| {
                    parse_error(
                        arm.line,
                        format!("`switch` case label must be non-negative, got `{value}`"),
                    )
                })?;
                if labels.contains(&value) {
                    return Err(parse_error(
                        arm.line,
                        format!("`switch` case label {value} appears twice"),
                    ));
                }
                labels.push(value);
                values.push(value);
            }
            cases.push((values, body));
        }

        // The chain is exclusive only because no arm writes the switched
        // register, which is checked rather than assumed.
        let probe = ClassicalCondition::RegisterEquals {
            offset,
            size,
            value: 0,
        };
        let writes_operand = cases
            .iter()
            .map(|(_, body)| body)
            .chain(default.iter())
            .any(|body| crate::circuit::body_writes_condition_bits(body, &probe));
        if writes_operand {
            return Err(parse_error(
                line,
                "`switch` needs an operand no arm overwrites; \
                          an arm here measures into the switched register",
            ));
        }

        let mut out = Vec::new();
        for (values, body) in &cases {
            for &value in values {
                let condition = ClassicalCondition::RegisterEquals {
                    offset,
                    size,
                    value,
                };
                out.extend(guarded(condition, body.clone()));
            }
        }
        if let Some(body) = default {
            out.extend(nest_default_arm(
                offset,
                size,
                &labels,
                body,
                self.region_depth,
                line,
            )?);
        }
        Ok(out)
    }

    /// The one arm a `switch` on a parse-time value runs, or its `default`.
    fn exec_switch_constant(
        &mut self,
        value: ClassicalValue,
        arms: &[ast::SwitchArm<'a>],
        line: usize,
    ) -> Result<Vec<Instruction>> {
        if matches!(value, ClassicalValue::Float(float) if float.fract() != 0.0) {
            return Err(parse_error(
                line,
                format!("`switch` needs an integer, got {}", value.to_float()),
            ));
        }
        let value = value.to_int();
        let mut labels: Vec<i128> = Vec::new();
        let mut taken = None;
        let mut default = None;
        for arm in arms {
            let Some(arm_labels) = &arm.labels else {
                if default.replace(&arm.body).is_some() {
                    return Err(parse_error(
                        arm.line,
                        "`switch` has more than one `default` arm",
                    ));
                }
                continue;
            };
            for label in arm_labels {
                let label = self.int_of(label, arm.line)?;
                if labels.contains(&label) {
                    return Err(parse_error(
                        arm.line,
                        format!("`switch` case label {label} appears twice"),
                    ));
                }
                labels.push(label);
                if label == value {
                    taken = Some(&arm.body);
                }
            }
        }
        taken
            .or(default)
            .map_or(Ok(Vec::new()), |body| self.exec_box(body))
    }

    /// The bit an unsubscripted name stands for when it names exactly one,
    /// as `bit b;` or a `def`'s `bit` parameter does.
    fn single_bit(&self, operand: &Operand) -> Option<usize> {
        if operand.index.is_some() {
            return None;
        }
        match self.bits_of(operand).ok()?.as_slice() {
            [bit] => Some(*bit),
            _ => None,
        }
    }

    pub(in crate::circuit::openqasm) fn condition_of(
        &self,
        condition: &Condition,
        line: usize,
    ) -> Result<ClassicalCondition> {
        match condition {
            Condition::Truthy(operand) => {
                if let Some(bit) = self.single_bit(operand) {
                    return Ok(ClassicalCondition::BitIsOne(bit));
                }
                if operand.index.is_none() {
                    return Err(parse_error(
                        line,
                        format!(
                            "expected `{0}==value`, `{0}!=value`, `{0}[i]`, `!{0}[i]` or \
                             `{0}[i]==0/1` in `if` condition, got `{0}`",
                            operand.describe()
                        ),
                    ));
                }
                Ok(ClassicalCondition::BitIsOne(self.bit_of(operand)?))
            }
            Condition::Negated(operand) => {
                if let Some(bit) = self.single_bit(operand) {
                    return Ok(ClassicalCondition::BitIsZero(bit));
                }
                if operand.index.is_none() {
                    return Err(parse_error(
                        line,
                        format!(
                            "expected `!c[i]` form in `if` condition, got `!{}`",
                            operand.describe()
                        ),
                    ));
                }
                Ok(ClassicalCondition::BitIsZero(self.bit_of(operand)?))
            }
            Condition::Parity { bits, compare } => {
                let expected = match compare {
                    None => true,
                    Some((op, rhs)) => {
                        let negate = *op == CmpOp::NotEqual;
                        match self.integer_of(rhs, line)? {
                            0 => negate,
                            1 => !negate,
                            other => {
                                return Err(parse_error(
                                    line,
                                    format!("a parity compares against 0 or 1, got `{other}`"),
                                ));
                            }
                        }
                    }
                };
                let mut resolved: Vec<usize> = Vec::with_capacity(bits.len());
                for bit in bits {
                    if bit.index.is_none() {
                        return Err(parse_error(
                            line,
                            format!("`{}` is a register where a bit belongs", bit.describe()),
                        ));
                    }
                    resolved.push(self.bit_of(bit)?);
                }
                Ok(ClassicalCondition::Parity {
                    bits: resolved.into_boxed_slice(),
                    expected,
                })
            }
            Condition::Expr(expr) => {
                guard_condition(&self.lower_expr(expr, line)?).ok_or_else(|| {
                    unsupported(
                        line,
                        format!(
                            "condition `{expr}`, which no guard on classical bits spells; \
                             `openqasm::parse_dynamic` evaluates it at runtime"
                        ),
                    )
                })
            }
            Condition::Compare { lhs, op, rhs } => {
                let negate = *op == CmpOp::NotEqual;
                let value = self.integer_of(rhs, line)?;
                if value < 0 {
                    return Err(parse_error(
                        line,
                        format!("negative integer in `if` condition is not supported: `{value}`"),
                    ));
                }
                let value = value as u64;
                if lhs.index.is_some() {
                    let bit = self.bit_of(lhs)?;
                    return Ok(match (value, negate) {
                        (0, false) | (1, true) => ClassicalCondition::BitIsZero(bit),
                        (1, false) | (0, true) => ClassicalCondition::BitIsOne(bit),
                        (other, _) => {
                            return Err(parse_error(
                                line,
                                format!("bit comparison must be against 0 or 1, got `{other}`"),
                            ));
                        }
                    });
                }
                let name = lhs
                    .register()
                    .ok_or_else(|| PrismError::UndefinedRegister {
                        name: lhs.describe(),
                        line,
                    })?;
                let register =
                    self.cregs
                        .get(name)
                        .ok_or_else(|| PrismError::UndefinedRegister {
                            name: name.to_string(),
                            line,
                        })?;
                Ok(if negate {
                    ClassicalCondition::RegisterNotEquals {
                        offset: register.offset,
                        size: register.size,
                        value,
                    }
                } else {
                    ClassicalCondition::RegisterEquals {
                        offset: register.offset,
                        size: register.size,
                        value,
                    }
                })
            }
        }
    }
}
