//! Walking a parsed program into instructions.
//!
//! Every name a statement reads is resolved against the scope the evaluator
//! holds, so a `for` body or a `gate` body is parsed once and run against
//! fresh bindings rather than rewritten and reparsed.

use std::collections::HashMap;

use super::*;
use crate::circuit::qasm::ast::{
    self, Argument, AssignOp, Block, CmpOp, Condition, DefParam, ForRange, Index, Operand,
    OperandName, Stmt, StmtKind,
};
use crate::circuit::qasm::expr::{self as syntax_expr, BinaryOp, Duration, Expr, Timed};
use crate::circuit::qasm::{lexer, parser as syntax};

/// Highest `$k` the program names, `None` when it names none.
///
/// Physical qubits declare no register, so the width comes from the tree
/// before any of it runs.
fn highest_physical(block: &Block) -> Option<usize> {
    let mut highest: Option<usize> = None;
    let mut note = |operand: &Operand| {
        if let OperandName::Physical(index) = operand.name {
            highest = Some(highest.map_or(index, |seen: usize| seen.max(index)));
        }
    };
    walk_operands(block, &mut note);
    highest
}

fn walk_operands(block: &Block, note: &mut impl FnMut(&Operand)) {
    for stmt in block {
        match &stmt.kind {
            StmtKind::Alias { sources, .. } => sources.iter().for_each(&mut *note),
            StmtKind::Call {
                params, operands, ..
            } => {
                for param in params {
                    if let Argument::Operand(operand) = param {
                        note(operand);
                    }
                }
                operands.iter().for_each(&mut *note);
            }
            StmtKind::Measure(measure) => {
                note(&measure.source);
                note(&measure.target);
            }
            StmtKind::CallAssign(assign) => {
                note(&assign.target);
                for arg in &assign.args {
                    if let Argument::Operand(operand) = arg {
                        note(operand);
                    }
                }
            }
            StmtKind::Return(Some(ast::ReturnValue::Measure(source))) => note(source),
            StmtKind::Reset { targets }
            | StmtKind::Barrier { targets }
            | StmtKind::Delay { targets, .. } => targets.iter().for_each(&mut *note),
            StmtKind::If(conditional) => {
                match &conditional.condition {
                    Condition::Truthy(operand) | Condition::Negated(operand) => note(operand),
                    Condition::Compare { lhs, .. } => note(lhs),
                    Condition::Parity { bits, .. } => bits.iter().for_each(&mut *note),
                }
                walk_operands(&conditional.then_body, note);
                if let Some(body) = &conditional.else_body {
                    walk_operands(body, note);
                }
            }
            StmtKind::For { body, .. }
            | StmtKind::Box { body, .. }
            | StmtKind::DefDef { body, .. }
            | StmtKind::GateDef { body, .. } => walk_operands(body, note),
            StmtKind::Switch { operand, arms } => {
                note(operand);
                for arm in arms {
                    walk_operands(&arm.body, note);
                }
            }
            _ => {}
        }
    }
}

/// Fold an expression to a number, where a ratio of two durations counts as
/// one. The duration path runs only once the plain fold has failed, so a
/// program that names no duration never pays for it.
fn number_of(
    expr: &Expr,
    line: usize,
    vars: Option<&HashMap<&str, f64>>,
    durations: &HashMap<&str, Timed>,
) -> Result<f64> {
    let value = syntax_expr::eval(expr, line, vars);
    if value.is_ok() || !syntax_expr::is_timed(expr, durations) {
        return value;
    }
    match syntax_expr::eval_timed(expr, line, vars, durations)? {
        Timed::Number(value) => Ok(value),
        Timed::Duration(_) => Err(parse_error(
            line,
            format!("`{expr}` is a duration where a number belongs"),
        )),
        Timed::Stretch => Err(parse_error(
            line,
            format!("`{expr}` is a stretch where a number belongs"),
        )),
    }
}

fn is_bit_param(args: &[DefParam], name: &str) -> bool {
    args.iter()
        .any(|arg| matches!(arg, DefParam::Bit { name: param, .. } if *param == name))
}

/// Line of the first `return` in `block`, nested bodies included.
fn return_line(block: &[Stmt]) -> Option<usize> {
    block.iter().find_map(|stmt| match &stmt.kind {
        StmtKind::Return(_) => Some(stmt.line),
        StmtKind::If(conditional) => return_line(&conditional.then_body)
            .or_else(|| conditional.else_body.as_deref().and_then(return_line)),
        StmtKind::For { body, .. } | StmtKind::Box { body, .. } => return_line(body),
        StmtKind::Switch { arms, .. } => arms.iter().find_map(|arm| return_line(&arm.body)),
        _ => None,
    })
}

impl<'a> Parser<'a> {
    /// Lex, parse, and run the source.
    pub(super) fn parse_program(
        &mut self,
    ) -> Result<(Circuit, Parameters, Vec<ResultSpec>, Option<NoiseModel>)> {
        let tokens = lexer::tokenize(self.input)?;
        let program = syntax::parse_program(&tokens)?;
        if let Some(highest) = highest_physical(&program) {
            self.physical = true;
            self.total_qubits = highest + 1;
        }
        let instructions = self.execute(&program)?;
        if self.verbatim_pending {
            return Err(parse_error(
                program.last().map_or(1, |stmt| stmt.line),
                "a verbatim pragma must be followed by a `box`",
            ));
        }

        let circuit = Circuit {
            num_qubits: self.total_qubits,
            num_classical_bits: self.total_cbits,
            instructions,
        };
        let params =
            Parameters::from_links(std::mem::take(&mut self.links), self.input_names.len())
                .with_names(std::mem::take(&mut self.input_names))
                .pinned_to(&circuit);
        let noise = self.build_noise_model(&circuit)?;
        Ok((circuit, params, std::mem::take(&mut self.results), noise))
    }

    /// Run a block, threading the parameter link and noise event each statement
    /// may leave behind onto the instruction index they belong to.
    pub(super) fn execute(&mut self, block: &Block<'a>) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::with_capacity(block.len());
        for stmt in block {
            let base = instructions.len();
            self.pending_input_slot = None;
            self.exec_stmt(stmt, &mut instructions)?;
            if let Some(slot) = self.pending_input_slot.take() {
                for offset in 0..instructions.len() - base {
                    self.links.push(ParamLink {
                        instruction: base + offset,
                        slot,
                    });
                }
            }
            if let Some(spec) = self.pending_noise.take() {
                // A channel acts at the point the pragma stands, which is after
                // everything emitted so far.
                if base == 0 {
                    return Err(parse_error(
                        stmt.line,
                        "a noise pragma needs a preceding instruction to follow",
                    ));
                }
                self.noise_specs.push((base - 1, spec));
            }
        }
        Ok(instructions)
    }

    fn exec_stmt(&mut self, stmt: &Stmt<'a>, out: &mut Vec<Instruction>) -> Result<()> {
        let line = stmt.line;
        match &stmt.kind {
            StmtKind::Empty | StmtKind::Include => Ok(()),
            StmtKind::Version(version) => {
                Self::check_version_number(version, line)?;
                Ok(())
            }
            StmtKind::RegisterDecl { kind, name, size } => {
                if let Some(result) = self.def_result.as_ref().filter(|result| {
                    *kind == ast::RegisterKind::Classical && result.local == Some(*name)
                }) {
                    // The returned bit is the caller's target, bound before the
                    // body ran, so its declaration allocates nothing.
                    let width = match size {
                        Some(size) => self.integer_of(size, line)?,
                        None => 1,
                    };
                    if width != result.bits.len() as i64 {
                        return Err(parse_error(
                            line,
                            format!(
                                "`{name}` declares {width} bit(s) where the def returns {}",
                                result.bits.len()
                            ),
                        ));
                    }
                    return Ok(());
                }
                self.declare_register(*kind, name, size.as_ref(), line)?;
                Ok(())
            }
            StmtKind::InputDecl { ty, name } => {
                self.declare_input(ty, name, line)?;
                Ok(())
            }
            StmtKind::OutputDecl { ty, name, size } => {
                if *ty != "bit" {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!("`output {ty}` (only bit outputs are reported)"),
                        line,
                    });
                }
                self.declare_register(ast::RegisterKind::Classical, name, size.as_ref(), line)?;
                Ok(())
            }
            StmtKind::ClassicalDecl {
                constant,
                ty,
                name,
                value,
            } => {
                self.declare_classical(*constant, ty, name, value.as_ref(), line)?;
                Ok(())
            }
            StmtKind::Assign { target, op, value } => {
                self.assign_classical(target, *op, value, line)?;
                Ok(())
            }
            StmtKind::Alias { name, sources } => {
                self.declare_alias(name, sources, line)?;
                Ok(())
            }
            StmtKind::Call {
                modifiers,
                name,
                params,
                operands,
            } => self.exec_call(modifiers, name, params, operands, line, out),
            StmtKind::Measure(measure) => {
                let qubits = self.qubits_of(&measure.source)?;
                let bits = self.bits_of(&measure.target)?;
                out.extend(Self::build_measurements(qubits, bits, line)?);
                Ok(())
            }
            StmtKind::Reset { targets } => {
                for target in targets {
                    out.extend(
                        self.qubits_of(target)?
                            .into_iter()
                            .map(|qubit| Instruction::Reset { qubit }),
                    );
                }
                Ok(())
            }
            StmtKind::Barrier { targets } => {
                let mut qubits = SmallVec::<[usize; 4]>::new();
                if targets.is_empty() {
                    qubits.extend(0..self.total_qubits);
                } else {
                    for target in targets {
                        qubits.extend(self.qubits_of(target)?);
                    }
                }
                out.push(Instruction::Barrier { qubits });
                Ok(())
            }
            // An ideal simulation has no clock, so a delay is the identity on
            // its qubits. It emits nothing, which also leaves fusion free to
            // merge across it.
            StmtKind::Delay { duration, targets } => {
                self.span_of(duration, "delay", line)?;
                for target in targets {
                    self.qubits_of(target)?;
                }
                Ok(())
            }
            StmtKind::If(conditional) => {
                out.extend(self.exec_if(conditional, line)?);
                Ok(())
            }
            StmtKind::For {
                variable,
                range,
                body,
            } => {
                out.extend(self.exec_for(variable, range, body, line)?);
                Ok(())
            }
            StmtKind::Switch { operand, arms } => {
                out.extend(self.exec_switch(operand, arms, line)?);
                Ok(())
            }
            StmtKind::GateDef {
                name,
                params,
                qubits,
                body,
            } => {
                if body.is_empty() {
                    return Err(parse_error(
                        line,
                        format!("gate '{name}' has an empty body"),
                    ));
                }
                self.gate_defs.insert(
                    *name,
                    GateDefinition {
                        params: params.clone(),
                        qubits: qubits.clone(),
                        body: body.clone(),
                    },
                );
                Ok(())
            }
            StmtKind::DefDef {
                name,
                args,
                result,
                body,
            } => {
                self.declare_def(name, args, result.as_ref(), body, line)?;
                Ok(())
            }
            StmtKind::Return(value) => {
                let Some(result) = &self.def_result else {
                    return Err(PrismError::UnsupportedConstruct {
                        construct: "`return` outside a def".to_string(),
                        line,
                    });
                };
                if let Some(ast::ReturnValue::Measure(source)) = value {
                    let qubits = self.qubits_of(source)?;
                    out.extend(Self::build_measurements(qubits, result.bits.clone(), line)?);
                }
                Ok(())
            }
            StmtKind::CallAssign(assign) => self.exec_call_assign(assign, line, out),
            StmtKind::Box { duration, body } => {
                // A box and Braket's verbatim pragma both direct a scheduler or
                // a device compiler, which a simulator has nothing to honour.
                self.verbatim_pending = false;
                if let Some(duration) = duration {
                    self.span_of(duration, "box", line)?;
                }
                out.extend(self.exec_box(body)?);
                Ok(())
            }
            StmtKind::Pragma(text) => {
                out.extend(self.exec_pragma(text, line)?);
                Ok(())
            }
        }
    }

    // ---------------------------------------------------------- declarations

    fn declare_register(
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
        match kind {
            ast::RegisterKind::Qubit => {
                self.reject_mixed_addressing(line)?;
                let offset = self.total_qubits;
                self.total_qubits += width;
                self.qregs.insert(
                    name,
                    Register {
                        offset,
                        size: width,
                    },
                );
            }
            ast::RegisterKind::Classical => {
                let offset = self.total_cbits;
                self.total_cbits += width;
                self.cregs.insert(
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

    fn declare_input(&mut self, ty: &str, name: &'a str, line: usize) -> Result<()> {
        if !matches!(ty, "float" | "angle") {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("`input {ty}` (only float and angle inputs bind to an angle)"),
                line,
            });
        }
        if self.inputs.contains_key(name) {
            return Err(parse_error(line, format!("input `{name}` declared twice")));
        }
        self.inputs.insert(name, self.input_names.len());
        self.input_names.push(name.to_string());
        Ok(())
    }

    fn declare_classical(
        &mut self,
        constant: bool,
        ty: &str,
        name: &'a str,
        value: Option<&Expr<'a>>,
        line: usize,
    ) -> Result<()> {
        let kind = match ty {
            "int" | "uint" => ClassicalType::Int,
            "bool" => ClassicalType::Bool,
            "float" | "angle" => ClassicalType::Float,
            "duration" => ClassicalType::Duration,
            "stretch" => ClassicalType::Stretch,
            other => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("`{other}` declarations"),
                    line,
                });
            }
        };
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
                self.bind_duration(name, kind, zero, constant);
            } else {
                self.bind_classical(name, kind, 0.0, constant);
            }
            return Ok(());
        };
        if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
            let folded = self.duration_of(value, line)?;
            self.bind_duration(name, kind, folded, constant);
            return Ok(());
        }
        let folded = self.fold_typed(kind, value, line)?;
        self.bind_classical(name, kind, folded, constant);
        Ok(())
    }

    /// Bind a `duration` or `stretch`. A stretch keeps no value: a scheduler
    /// sizes it, so whatever it was initialised with is only a lower bound.
    fn bind_duration(&mut self, name: &'a str, ty: ClassicalType, value: Timed, constant: bool) {
        let value = if ty == ClassicalType::Stretch {
            Timed::Stretch
        } else {
            value
        };
        self.classical.insert(name, ClassicalDecl { ty, constant });
        self.durations.insert(name, value);
    }

    /// Fold an expression that has to be a duration or a stretch.
    fn duration_of(&self, expr: &Expr, line: usize) -> Result<Timed> {
        match syntax_expr::eval_timed(expr, line, self.param_vars.as_ref(), &self.durations)? {
            Timed::Number(value) => Err(parse_error(
                line,
                format!("`{expr}` is the number {value} where a duration belongs; give it a unit"),
            )),
            timed => Ok(timed),
        }
    }

    /// Check the length a `delay` or `box` names, which has no further effect.
    fn span_of(&self, expr: &Expr, what: &str, line: usize) -> Result<()> {
        if let Timed::Duration(duration) = self.duration_of(expr, line)? {
            if duration.is_negative() {
                return Err(parse_error(
                    line,
                    format!("`{what}` needs a non-negative duration, got `{expr}`"),
                ));
            }
        }
        Ok(())
    }

    fn assign_classical(
        &mut self,
        target: &'a str,
        op: Option<AssignOp>,
        value: &Expr<'a>,
        line: usize,
    ) -> Result<()> {
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
        if matches!(kind, ClassicalType::Duration | ClassicalType::Stretch) {
            return self.assign_duration(target, kind, op, value, line);
        }
        let folded = self.fold_typed(kind, value, line)?;
        let updated = match op {
            None => folded,
            Some(op) => {
                let current = self.fold_typed(kind, &Expr::Ident(target), line)?;
                match op {
                    AssignOp::Add => current + folded,
                    AssignOp::Sub => current - folded,
                    AssignOp::Mul => current * folded,
                    AssignOp::Div if folded == 0.0 => {
                        return Err(parse_error(
                            line,
                            "division by zero in a compound assignment",
                        ));
                    }
                    AssignOp::Rem if folded == 0.0 => {
                        return Err(parse_error(line, "modulo by zero in a compound assignment"));
                    }
                    AssignOp::Div => current / folded,
                    AssignOp::Rem => current % folded,
                }
            }
        };
        self.bind_classical(target, kind, updated, false);
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
        let value =
            syntax_expr::eval_timed(value, line, self.param_vars.as_ref(), &self.durations)?;
        let updated = match op {
            None => value,
            Some(op) => {
                let op = match op {
                    AssignOp::Add => BinaryOp::Add,
                    AssignOp::Sub => BinaryOp::Sub,
                    AssignOp::Mul => BinaryOp::Mul,
                    AssignOp::Div => BinaryOp::Div,
                    AssignOp::Rem => BinaryOp::Rem,
                };
                let current = self
                    .durations
                    .get(target)
                    .copied()
                    .unwrap_or(Timed::Stretch);
                syntax_expr::combine_timed(op, current, value, line)?
            }
        };
        if let Timed::Number(number) = updated {
            return Err(parse_error(
                line,
                format!("assignment leaves duration `{target}` holding the number {number}"),
            ));
        }
        self.bind_duration(target, kind, updated, false);
        Ok(())
    }

    /// Fold an initializer to the value its declared type holds, naming an
    /// `input` rather than leaving it to read as an unknown identifier.
    fn fold_typed(&self, kind: ClassicalType, value: &Expr, line: usize) -> Result<f64> {
        if let Some(name) = self.input_names.iter().find(|name| value.mentions(name)) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "input `{name}` in a classical declaration; an input binds an angle whole"
                ),
                line,
            });
        }
        match kind {
            ClassicalType::Float | ClassicalType::Duration | ClassicalType::Stretch => {
                self.value_of(value, line)
            }
            ClassicalType::Int => Ok(self.integer_of(value, line)? as f64),
            ClassicalType::Bool => Ok(f64::from(self.integer_of(value, line)? != 0)),
        }
    }

    fn declare_alias(&mut self, name: &'a str, sources: &[Operand<'a>], line: usize) -> Result<()> {
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
        self.aliases.insert(name, Alias { kind, indices });
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
        let alias_kind = self.aliases.get(name).map(|alias| alias.kind);
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

    fn declare_def(
        &mut self,
        name: &'a str,
        args: &[DefParam<'a>],
        result: Option<&ast::BitResult<'a>>,
        body: &Block<'a>,
        line: usize,
    ) -> Result<()> {
        let Some((last, rest)) = body.split_last() else {
            return Err(parse_error(line, format!("def `{name}` has an empty body")));
        };
        // A `return` anywhere but last decides at run time whether the rest of
        // the body runs, which no guarded region can say.
        let early = return_line(rest).or_else(|| match last.kind {
            StmtKind::Return(_) => None,
            _ => return_line(std::slice::from_ref(last)),
        });
        if let Some(at) = early {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "`return` before the end of def `{name}`, which needs control flow \
                     beyond a guarded region"
                ),
                line: at,
            });
        }
        let local = Self::def_result_local(name, args, result.is_some(), last, rest)?;
        Self::check_def_writes(name, args, local, body, true)?;
        self.def_defs.insert(
            name,
            DefDefinition {
                args: args.to_vec(),
                result: result.cloned(),
                local,
                body: body.clone(),
            },
        );
        Ok(())
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

    /// Check that a `def` body writes classical bits only through its result.
    ///
    /// A bit parameter is a copy, and any other bit would need a classical
    /// register the program never declared, so neither can be written here.
    fn check_def_writes(
        name: &str,
        args: &[DefParam<'a>],
        local: Option<&str>,
        block: &Block<'a>,
        top: bool,
    ) -> Result<()> {
        for stmt in block {
            match &stmt.kind {
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
                    Self::check_def_writes(name, args, local, &conditional.then_body, false)?;
                    if let Some(body) = &conditional.else_body {
                        Self::check_def_writes(name, args, local, body, false)?;
                    }
                }
                StmtKind::For { body, .. } | StmtKind::Box { body, .. } => {
                    Self::check_def_writes(name, args, local, body, false)?;
                }
                StmtKind::Switch { arms, .. } => {
                    for arm in arms {
                        Self::check_def_writes(name, args, local, &arm.body, false)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------- operands

    /// Qubits an operand names: a physical index, a whole register, one
    /// element, a slice, or an alias of any of those.
    pub(super) fn qubits_of(&self, operand: &Operand) -> Result<SmallVec<[usize; 4]>> {
        let name = match &operand.name {
            OperandName::Physical(index) => return Ok(smallvec![*index]),
            OperandName::Register(name) => name,
        };
        if let Some(indices) = self.alias_indices(ast::RegisterKind::Qubit, name, operand)? {
            return Ok(indices);
        }
        self.register_indices(&self.qregs, ast::RegisterKind::Qubit, name, operand)
    }

    fn bits_of(&self, operand: &Operand) -> Result<SmallVec<[usize; 4]>> {
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

    fn bit_of(&self, operand: &Operand) -> Result<usize> {
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
        let Some(alias) = self.aliases.get(name) else {
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
    fn pick_indices(
        &self,
        size: usize,
        index: &Index,
        kind: ast::RegisterKind,
        operand: &Operand,
    ) -> Result<SmallVec<[usize; 4]>> {
        let line = operand.line;
        let bound = |value: i64| -> Result<usize> {
            let at = usize::try_from(value).map_err(|_| {
                parse_error(line, format!("negative index in `{}`", operand.describe()))
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
                let first = at(&range.start, 0)?;
                let last = at(&range.stop, size as i64 - 1)?;
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
                    at += stride;
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

    // ------------------------------------------------------------ expressions

    pub(super) fn value_of(&self, expr: &Expr, line: usize) -> Result<f64> {
        number_of(expr, line, self.param_vars.as_ref(), &self.durations)
    }

    pub(super) fn integer_of(&self, expr: &Expr, line: usize) -> Result<i64> {
        let value = self.value_of(expr, line)?;
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(parse_error(
                line,
                format!("expected an integer, got {value}"),
            ));
        }
        if value > i64::MAX as f64 || value < i64::MIN as f64 {
            return Err(parse_error(line, "integer expression out of range"));
        }
        Ok(value as i64)
    }

    // ------------------------------------------------------------------ calls

    fn exec_call(
        &mut self,
        modifiers: &[ast::Modifier],
        name: &str,
        params: &[Argument],
        operands: &[Operand],
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        let modifiers = self.fold_modifiers(modifiers, line)?;

        if self.def_defs.contains_key(name) {
            let instrs = self.expand_def(name, params, None, line)?;
            out.extend(Self::modify_expansion(instrs, &modifiers, name, line)?);
            return Ok(());
        }

        let (values, input_slot) = self.fold_params(params, line)?;
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
        if input_slot.is_some() && !modifiers.is_empty() {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("modifier on `{name}` reading an `input`"),
                line,
            });
        }

        let resolved: SmallVec<[SmallVec<[usize; 4]>; 4]> = operands
            .iter()
            .map(|operand| self.qubits_of(operand))
            .collect::<Result<_>>()?;
        let width = self.broadcast_length(&resolved, name, line)?;

        let base = out.len();
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
                name,
                &values,
                &modifiers,
                &qubits,
                input_slot.is_some(),
                line,
                out,
            )?;
        }
        if input_slot.is_some() {
            Self::check_every_instruction_is_bindable(&out[base..], name, line)?;
        }
        self.pending_input_slot = input_slot;
        Ok(())
    }

    fn fold_modifiers(
        &self,
        modifiers: &[ast::Modifier],
        line: usize,
    ) -> Result<Vec<super::Modifier>> {
        modifiers
            .iter()
            .map(|modifier| {
                Ok(match modifier {
                    ast::Modifier::Inv => super::Modifier::Inv,
                    ast::Modifier::Ctrl { negated } => super::Modifier::Ctrl { negated: *negated },
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
                        super::Modifier::Pow(exponent)
                    }
                })
            })
            .collect()
    }

    /// Evaluate a call's angle arguments, reporting the one `input` slot they
    /// read. Two inputs on one gate are rejected: a slot is written onto the
    /// single angle a bindable gate carries.
    fn fold_params(&self, params: &[Argument], line: usize) -> Result<(Vec<f64>, Option<usize>)> {
        let mut values = Vec::with_capacity(params.len());
        let mut input_slot = None;
        for param in params {
            let Argument::Value(expr) = param else {
                return Err(parse_error(
                    line,
                    "a gate angle cannot be a qubit reference",
                ));
            };
            let (value, slot) = self.fold_angle(expr, line)?;
            if slot.is_some() && input_slot.is_some() {
                return Err(PrismError::UnsupportedConstruct {
                    construct: "two `input` parameters on one gate".to_string(),
                    line,
                });
            }
            input_slot = input_slot.or(slot);
            values.push(value);
        }
        Ok((values, input_slot))
    }

    /// One angle argument, reporting the slot when it is exactly an `input`.
    fn fold_angle(&self, expr: &Expr, line: usize) -> Result<(f64, Option<usize>)> {
        if let Some((name, &slot)) = expr
            .as_ident()
            .and_then(|name| self.inputs.get(name).map(|slot| (name, slot)))
        {
            if self.nested {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!("input `{name}` inside a block body"),
                    line,
                });
            }
            return Ok((0.0, Some(slot)));
        }
        if let Some(name) = self.input_names.iter().find(|name| expr.mentions(name)) {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!(
                    "an expression over input `{name}`; an input binds an angle whole"
                ),
                line,
            });
        }
        Ok((self.value_of(expr, line)?, None))
    }

    /// Expand a user `gate` by binding its qubit names to the call's qubits and
    /// its parameters to the call's values, then running the body.
    pub(super) fn expand_user_gate(
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
        let bindings: HashMap<&'a str, f64> = def
            .params
            .iter()
            .copied()
            .zip(call_params.iter().copied())
            .collect();
        let mut sub = self.expansion_parser(bindings);
        for (&qubit_name, qubit) in def.qubits.iter().zip(call_qubits) {
            sub.aliases.insert(
                qubit_name,
                Alias {
                    kind: ast::RegisterKind::Qubit,
                    indices: vec![*qubit],
                },
            );
        }
        Ok(Some(sub.execute(&def.body)?))
    }

    /// `target = f(...);`: a `def` call whose `bit` result lands on `target`,
    /// or a builtin on the right of a classical assignment.
    fn exec_call_assign(
        &mut self,
        assign: &ast::CallAssign<'a>,
        line: usize,
        out: &mut Vec<Instruction>,
    ) -> Result<()> {
        if self.def_defs.contains_key(assign.name) {
            let bits = self.bits_of(&assign.target)?;
            out.extend(self.expand_def(assign.name, &assign.args, Some(&bits), line)?);
            return Ok(());
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
        let args = assign
            .args
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

    /// Inline a `def` call, binding each argument by the kind its declaration
    /// gave it and the result, when there is one, to `result`.
    fn expand_def(
        &self,
        name: &str,
        args: &[Argument],
        result: Option<&[usize]>,
        line: usize,
    ) -> Result<Vec<Instruction>> {
        if self.gate_expansion_depth >= MAX_GATE_EXPANSION_DEPTH {
            return Err(parse_error(
                line,
                format!(
                    "def expansion depth exceeds maximum ({MAX_GATE_EXPANSION_DEPTH}); \
                     possible recursive call to `{name}`"
                ),
            ));
        }
        let def = self.def_defs.get(name).expect("checked by the caller");
        if args.len() != def.args.len() {
            return Err(PrismError::GateArity {
                gate: name.to_string(),
                expected: def.args.len(),
                got: args.len(),
            });
        }

        let mut bindings: HashMap<&'a str, f64> = self.param_vars.clone().unwrap_or_default();
        let mut durations = self.durations.clone();
        let mut qubit_bindings: Vec<(&'a str, usize)> = Vec::new();
        let mut bit_bindings: Vec<(&'a str, SmallVec<[usize; 4]>)> = Vec::new();
        for (slot, argument) in def.args.iter().zip(args) {
            match slot {
                DefParam::Qubit(param_name) => {
                    let param_name = *param_name;
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
                    match self.qubits_of(&operand)?.as_slice() {
                        [single] => qubit_bindings.push((param_name, *single)),
                        _ => {
                            return Err(parse_error(
                                line,
                                format!(
                                    "def `{name}` qubit parameter `{param_name}` requires a \
                                     single qubit, got register `{}`",
                                    operand.describe()
                                ),
                            ));
                        }
                    }
                }
                DefParam::Value {
                    name: param_name,
                    integral,
                } => {
                    let param_name = *param_name;
                    let Argument::Value(expr) = argument else {
                        return Err(parse_error(
                            line,
                            format!("def `{name}` parameter `{param_name}` takes a value"),
                        ));
                    };
                    // The body reads the argument's value, so an input bound
                    // later would never reach it.
                    if let Some(input) = self.input_names.iter().find(|input| expr.mentions(input))
                    {
                        return Err(PrismError::UnsupportedConstruct {
                            construct: format!("input `{input}` as an argument to `def {name}`"),
                            line,
                        });
                    }
                    let value = number_of(expr, line, Some(&bindings), &durations)?;
                    let value = if *integral {
                        let rounded = value.round();
                        if (value - rounded).abs() > 0.0 {
                            return Err(parse_error(
                                line,
                                format!("def `{name}` parameter `{param_name}` takes an integer"),
                            ));
                        }
                        rounded
                    } else {
                        value
                    };
                    bindings.insert(param_name, value);
                }
                DefParam::Duration(param_name) => {
                    let Argument::Value(expr) = argument else {
                        return Err(parse_error(
                            line,
                            format!("def `{name}` parameter `{param_name}` takes a duration"),
                        ));
                    };
                    let value = syntax_expr::eval_timed(expr, line, Some(&bindings), &durations)?;
                    if let Timed::Number(number) = value {
                        return Err(parse_error(
                            line,
                            format!(
                                "def `{name}` parameter `{param_name}` takes a duration, got {number}"
                            ),
                        ));
                    }
                    durations.insert(param_name, value);
                }
                DefParam::Bit {
                    name: param_name,
                    width,
                } => {
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
                    bit_bindings.push((param_name, bits));
                }
            }
        }

        let frame = match (&def.result, result) {
            (None, None) => DefResult {
                bits: SmallVec::new(),
                local: None,
            },
            (None, Some(_)) => {
                return Err(parse_error(
                    line,
                    format!("def `{name}` declares no result to assign"),
                ));
            }
            (Some(_), None) => {
                return Err(PrismError::UnsupportedConstruct {
                    construct: format!(
                        "a call to def `{name}` that drops its `bit` result, which would need \
                         a classical bit the program never declared"
                    ),
                    line,
                });
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
                    return Err(PrismError::UnsupportedConstruct {
                        construct: format!(
                            "a call to def `{name}` passing a bit it also assigns, which the \
                             body would read after writing its result"
                        ),
                        line,
                    });
                }
                DefResult {
                    bits: bits.iter().copied().collect(),
                    local: def.local,
                }
            }
        };

        let mut sub = self.expansion_parser(bindings);
        sub.durations = durations;
        for (param_name, qubit) in qubit_bindings {
            sub.aliases.insert(
                param_name,
                Alias {
                    kind: ast::RegisterKind::Qubit,
                    indices: vec![qubit],
                },
            );
        }
        for (param_name, bits) in bit_bindings {
            sub.bind_bits(param_name, bits);
        }
        if let Some(local) = frame.local {
            sub.bind_bits(local, frame.bits.clone());
        }
        sub.def_result = Some(frame);
        sub.execute(&def.body)
    }

    /// Name `bits` inside an expanded body: as a register when they run
    /// contiguously, which a register comparison needs, and as an alias
    /// otherwise.
    fn bind_bits(&mut self, name: &'a str, bits: SmallVec<[usize; 4]>) {
        let contiguous = bits.windows(2).all(|pair| pair[1] == pair[0] + 1);
        match bits.first() {
            Some(&offset) if contiguous => {
                self.aliases.remove(name);
                self.cregs.insert(
                    name,
                    Register {
                        offset,
                        size: bits.len(),
                    },
                );
            }
            _ => {
                self.aliases.insert(
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
    fn expansion_parser(&self, bindings: HashMap<&'a str, f64>) -> Parser<'a> {
        let mut sub = Parser {
            dialect: self.dialect,
            input: "",
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            gate_defs: HashMap::new(),
            def_defs: HashMap::new(),
            total_qubits: self.total_qubits,
            total_cbits: self.total_cbits,
            gate_expansion_depth: self.gate_expansion_depth + 1,
            region_depth: self.region_depth,
            param_vars: Some(bindings),
            inputs: HashMap::new(),
            input_names: Vec::new(),
            links: Vec::new(),
            pending_input_slot: None,
            nested: true,
            results: Vec::new(),
            noise_specs: Vec::new(),
            pending_noise: None,
            verbatim_pending: false,
            physical: self.physical,
            aliases: HashMap::new(),
            classical: self.classical_copy(),
            durations: HashMap::new(),
            def_result: None,
        };
        for (name, register) in &self.qregs {
            sub.qregs.insert(
                *name,
                Register {
                    offset: register.offset,
                    size: register.size,
                },
            );
        }
        for (name, register) in &self.cregs {
            sub.cregs.insert(
                *name,
                Register {
                    offset: register.offset,
                    size: register.size,
                },
            );
        }
        for (name, alias) in &self.aliases {
            sub.aliases.insert(
                *name,
                Alias {
                    kind: alias.kind,
                    indices: alias.indices.clone(),
                },
            );
        }
        for (name, def) in &self.gate_defs {
            sub.gate_defs.insert(
                *name,
                GateDefinition {
                    params: def.params.clone(),
                    qubits: def.qubits.clone(),
                    body: def.body.clone(),
                },
            );
        }
        for (name, def) in &self.def_defs {
            sub.def_defs.insert(
                *name,
                DefDefinition {
                    args: def.args.clone(),
                    result: def.result.clone(),
                    local: def.local,
                    body: def.body.clone(),
                },
            );
        }
        sub
    }

    // ------------------------------------------------------------ control flow

    fn exec_if(
        &mut self,
        conditional: &ast::Conditional<'a>,
        line: usize,
    ) -> Result<Vec<Instruction>> {
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
    fn exec_box(&mut self, body: &Block<'a>) -> Result<Vec<Instruction>> {
        let before: Vec<&'a str> = self.classical.keys().copied().collect();
        let was_nested = std::mem::replace(&mut self.nested, true);
        let result = self.execute(body);
        self.nested = was_nested;
        let declared: Vec<&'a str> = self
            .classical
            .keys()
            .copied()
            .filter(|name| !before.contains(name))
            .collect();
        for name in declared {
            self.classical.remove(name);
            self.durations.remove(name);
            if let Some(values) = self.param_vars.as_mut() {
                values.remove(name);
            }
        }
        result
    }

    /// Run a block one nesting level down.
    fn region(&mut self, block: &Block<'a>) -> Result<Vec<Instruction>> {
        self.enter_region_depth(block)?;
        let was_nested = std::mem::replace(&mut self.nested, true);
        let body = self.execute(block);
        self.nested = was_nested;
        self.region_depth -= 1;
        body
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

    fn exec_for(
        &mut self,
        variable: &'a str,
        range: &ForRange<'a>,
        body: &Block<'a>,
        line: usize,
    ) -> Result<Vec<Instruction>> {
        let values = self.for_values(range, line)?;
        if values.len() as i64 > MAX_FOR_ITERATIONS {
            return Err(parse_error(
                line,
                format!(
                    "for loop iterates {} times (max {MAX_FOR_ITERATIONS})",
                    values.len()
                ),
            ));
        }
        let mut out = Vec::new();
        for value in values {
            // The body binds for one pass, so the whole classical scope is
            // restored rather than the loop variable alone.
            let saved_values = self.param_vars.clone();
            let saved_decls = self.classical_copy();
            let saved_durations = self.durations.clone();
            self.param_vars
                .get_or_insert_with(HashMap::new)
                .insert(variable, value as f64);
            let was_nested = std::mem::replace(&mut self.nested, true);
            let produced = self.execute(body);
            self.nested = was_nested;
            self.param_vars = saved_values;
            self.classical = saved_decls;
            self.durations = saved_durations;
            out.extend(produced?);
        }
        Ok(out)
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
                    at += stride;
                }
                Ok(values)
            }
        }
    }

    fn exec_switch(
        &mut self,
        operand: &Operand<'a>,
        arms: &[ast::SwitchArm<'a>],
        line: usize,
    ) -> Result<Vec<Instruction>> {
        let (offset, size) = self.switch_operand(operand)?;
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

    /// The `(offset, size)` of the classical range a `switch` reads. A bit
    /// reference is a range of one.
    fn switch_operand(&self, operand: &Operand) -> Result<(usize, usize)> {
        if operand.index.is_some() {
            return Ok((self.bit_of(operand)?, 1));
        }
        let name = operand
            .register()
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: operand.describe(),
                line: operand.line,
            })?;
        if let Some(alias) = self.aliases.get(name) {
            return match alias.indices.as_slice() {
                [single] => Ok((*single, 1)),
                _ => Err(parse_error(
                    operand.line,
                    format!("`switch` on alias `{name}`, which is not one contiguous register"),
                )),
            };
        }
        let register = self
            .cregs
            .get(name)
            .ok_or_else(|| PrismError::UndefinedRegister {
                name: name.to_string(),
                line: operand.line,
            })?;
        Ok((register.offset, register.size))
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

    fn condition_of(&self, condition: &Condition, line: usize) -> Result<ClassicalCondition> {
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

    // --------------------------------------------------------------- pragmas

    /// Dispatch a `#pragma braket ...` line.
    ///
    /// Only the Braket dialect reads these. Under any other the pragma is
    /// rejected rather than dropped, so a lost result request cannot pass
    /// unnoticed.
    fn exec_pragma(&mut self, text: &str, line: usize) -> Result<Vec<Instruction>> {
        let body = text.trim_start_matches("#pragma").trim();
        let Some(braket_body) = body.strip_prefix("braket") else {
            return Err(PrismError::UnsupportedConstruct {
                construct: format!("pragma `{body}`"),
                line,
            });
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
    pub(super) fn qubits_of_text(&self, token: &str, line: usize) -> Result<SmallVec<[usize; 4]>> {
        let mut out = SmallVec::new();
        for operand in syntax::parse_operands(&lexer::tokenize(token)?)? {
            out.extend(self.qubits_of(&Operand { line, ..operand })?);
        }
        Ok(out)
    }
}
