//! The pass that decides which classical variables live at runtime and which
//! branches need the control-flow graph.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use super::dynamic::{Decided, condition_expr, key, operand_expr};
use super::*;
use crate::circuit::dynamic;
use crate::circuit::qasm::ast::{Condition, Stmt, StmtKind};
use crate::circuit::qasm::expr::Expr;

#[cfg(test)]
thread_local! {
    /// Statements the runtime analysis has walked on this thread.
    pub(super) static WALKED: Cell<usize> = const { Cell::new(0) };
}

/// Which declarations live at runtime and which branches need the graph,
/// keyed by statement.
#[derive(Default)]
pub(super) struct RuntimeAnalysis {
    pub(super) runtime: HashSet<usize>,
    pub(super) branches: HashSet<usize>,
    /// `for` loops a measurement may break or continue.
    pub(super) unrolled: HashSet<usize>,
    pub(super) runtime_defs: Rc<HashSet<String>>,
    needs_call: bool,
}

#[derive(Clone, Copy)]
enum Binding {
    Var { decl: usize, depth: usize },
    Constant,
    Array { decl: usize, depth: usize },
    Bits,
    Other,
}

struct Walk<'s, 'a> {
    runtime: &'s mut HashSet<usize>,
    branches: HashSet<usize>,
    changed: bool,
    scopes: Vec<HashMap<&'a str, Binding>>,
    functions: &'s HashSet<String>,
    scalar_return: bool,
    needs_call: bool,
    local_bits: HashSet<&'a str>,
    initialized_bits: HashSet<&'a str>,
    single_bits: HashSet<&'a str>,
    early_bit_read: Cell<bool>,
    conditional: usize,
    /// Runtime depth of each enclosing loop body, innermost last.
    loops: Vec<usize>,
    /// `for` loops found to hold a measured exit, kept across passes of the
    /// fixed point so each body is walked once per pass.
    unrolled: &'s mut HashSet<usize>,
    /// The innermost `for` holds a `break` or `continue` a measurement decides.
    runtime_exit: bool,
}

#[derive(Clone, Copy, Default)]
struct Reads {
    runtime: bool,
    bits: bool,
}

impl Reads {
    fn any(self) -> bool {
        self.runtime || self.bits
    }

    fn or(self, other: Reads) -> Reads {
        Reads {
            runtime: self.runtime || other.runtime,
            bits: self.bits || other.bits,
        }
    }
}

impl RuntimeAnalysis {
    /// Iterate to a fixed point: a variable assigned from a runtime one is
    /// itself runtime, which can make a third one so on a later pass.
    pub(super) fn of(program: &[Stmt]) -> RuntimeAnalysis {
        let mut functions = Rc::new(HashSet::new());
        loop {
            let mut changed = false;
            for stmt in program {
                if let StmtKind::DefDef {
                    name,
                    args,
                    result,
                    body,
                } = &stmt.kind
                    && !functions.contains(*name)
                {
                    let needs_call = Self::for_def(
                        body,
                        args,
                        result.as_ref().is_some_and(|result| result.ty != "bit"),
                        Rc::clone(&functions),
                        &[],
                    )
                    .needs_call;
                    if needs_call {
                        Rc::make_mut(&mut functions).insert((*name).to_string());
                        changed = true;
                    }
                }
            }
            if !changed {
                return Self::for_def(program, &[], false, functions, &[]);
            }
        }
    }

    pub(super) fn for_def<'a>(
        program: &[Stmt<'a>],
        args: &[ast::DefParam<'a>],
        scalar_return: bool,
        functions: Rc<HashSet<String>>,
        runtime_params: &[&str],
    ) -> RuntimeAnalysis {
        let mut runtime = HashSet::new();
        let mut unrolled = HashSet::new();
        loop {
            let mut walk = Walk {
                runtime: &mut runtime,
                branches: HashSet::new(),
                changed: false,
                scopes: vec![HashMap::new()],
                functions: &functions,
                scalar_return,
                needs_call: false,
                local_bits: HashSet::new(),
                initialized_bits: HashSet::new(),
                single_bits: HashSet::new(),
                early_bit_read: Cell::new(false),
                conditional: 0,
                loops: Vec::new(),
                unrolled: &mut unrolled,
                runtime_exit: false,
            };
            for param in args {
                match param {
                    ast::DefParam::Bit { name, .. } => walk.bind(name, Binding::Bits),
                    ast::DefParam::Value { name, .. } => {
                        let decl = param as *const ast::DefParam as usize;
                        if runtime_params.contains(name) {
                            walk.mark(decl);
                        }
                        walk.bind(name, Binding::Var { decl, depth: 0 });
                    }
                    ast::DefParam::Duration(name)
                    | ast::DefParam::Qubit { name, .. }
                    | ast::DefParam::Array { name, .. } => {
                        walk.bind(name, Binding::Constant);
                    }
                }
            }
            walk.block(program, 0);
            if !walk.changed {
                let branches = walk.branches;
                let initializes_bits = walk.early_bit_read.get();
                let needs_call = walk.needs_call || initializes_bits || !runtime.is_empty();
                return RuntimeAnalysis {
                    runtime,
                    branches,
                    unrolled,
                    runtime_defs: functions,
                    needs_call,
                };
            }
        }
    }
}

impl<'a> Walk<'_, 'a> {
    fn bind(&mut self, name: &'a str, binding: Binding) {
        self.scopes
            .last_mut()
            .expect("a scope is always open")
            .insert(name, binding);
    }

    fn lookup(&self, name: &str) -> Binding {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
            .unwrap_or(Binding::Other)
    }

    fn mark(&mut self, decl: usize) {
        if self.runtime.insert(decl) {
            self.changed = true;
        }
    }

    fn name_reads(&self, name: &str) -> Reads {
        match self.lookup(name) {
            Binding::Var { decl, .. } | Binding::Array { decl, .. } => Reads {
                runtime: self.runtime.contains(&decl),
                bits: false,
            },
            Binding::Bits => {
                if self.local_bits.contains(name) && !self.initialized_bits.contains(name) {
                    self.early_bit_read.set(true);
                }
                Reads {
                    runtime: false,
                    bits: true,
                }
            }
            Binding::Constant | Binding::Other => Reads::default(),
        }
    }

    fn expr_reads(&self, expr: &Expr) -> Reads {
        with_stack(|| match expr {
            Expr::Runtime(_) => Reads {
                runtime: true,
                bits: false,
            },
            Expr::Number(_) | Expr::Complex(_) | Expr::Duration(_) | Expr::DurationOf(_) => {
                Reads::default()
            }
            Expr::Ident(name) => self.name_reads(name),
            Expr::Element(element) => element
                .indices
                .iter()
                .fold(self.name_reads(element.array), |reads, index| {
                    reads.or(self.expr_reads(index))
                }),
            Expr::Negate(inner) | Expr::Not(inner) | Expr::BitNot(inner) => self.expr_reads(inner),
            Expr::Binary { left, right, .. } => self.expr_reads(left).or(self.expr_reads(right)),
            Expr::Call(call) => call.args.iter().fold(
                Reads {
                    runtime: self.functions.contains(call.name),
                    bits: false,
                },
                |reads, arg| reads.or(self.expr_reads(arg)),
            ),
            Expr::Cast(cast) => self.expr_reads(&cast.value),
            Expr::Range(range) => range.bounds().fold(Reads::default(), |reads, bound| {
                reads.or(self.expr_reads(bound))
            }),
            Expr::Concat(parts) | Expr::Set(parts) => {
                parts.iter().fold(Reads::default(), |reads, part| {
                    reads.or(self.expr_reads(part))
                })
            }
        })
    }

    fn argument_reads(&self, argument: &ast::Argument) -> Reads {
        match argument {
            ast::Argument::Value(expr) => self.expr_reads(expr),
            ast::Argument::Operand(operand) => match operand_expr(operand) {
                Ok(expr) => self.expr_reads(&expr),
                Err(_) => Reads::default(),
            },
        }
    }

    fn decided(&self, condition: &Condition) -> Decided {
        let reads = match condition_expr(condition) {
            Ok(expr) => self.expr_reads(&expr),
            Err(_) => Reads {
                runtime: false,
                bits: true,
            },
        };
        Decided::classify(
            reads.runtime,
            reads.bits,
            matches!(condition, Condition::Expr(_)),
        )
    }

    /// A write to the variable or array `binding`, at runtime depth `depth`.
    fn write(&mut self, binding: Binding, depth: usize, reads: Reads) -> bool {
        match binding {
            Binding::Var { decl, depth: home } | Binding::Array { decl, depth: home } => {
                if depth > home || reads.any() {
                    self.mark(decl);
                }
                self.runtime.contains(&decl)
            }
            _ => false,
        }
    }

    fn scoped(&mut self, block: &[Stmt<'a>], depth: usize) -> bool {
        self.scopes.push(HashMap::new());
        let needs = self.block(block, depth);
        self.scopes.pop();
        needs
    }

    /// Walk `block` at runtime depth `depth`, reporting whether anything in it
    /// opens a block of the graph.
    fn block(&mut self, block: &[Stmt<'a>], depth: usize) -> bool {
        with_stack(|| {
            let mut needs = false;
            for stmt in block {
                needs |= self.stmt(stmt, depth);
            }
            needs
        })
    }

    fn stmt(&mut self, stmt: &Stmt<'a>, depth: usize) -> bool {
        #[cfg(test)]
        WALKED.with(|walked| walked.set(walked.get() + 1));
        match &stmt.kind {
            StmtKind::RegisterDecl {
                kind,
                name,
                size,
                init,
            } => {
                let binding = match kind {
                    ast::RegisterKind::Classical => Binding::Bits,
                    ast::RegisterKind::Qubit => Binding::Other,
                };
                self.bind(name, binding);
                if *kind == ast::RegisterKind::Classical {
                    self.local_bits.insert(name);
                    if size.is_none() || matches!(size, Some(Expr::Number(ClassicalValue::Int(1))))
                    {
                        self.single_bits.insert(name);
                    }
                    self.needs_call |= init.is_some();
                }
                match init.as_deref() {
                    Some(ast::BitInit::Bits(text)) => text.contains('1'),
                    Some(ast::BitInit::Value(value)) => !matches!(
                        value,
                        Expr::Number(
                            dynamic::ClassicalValue::Int(0) | dynamic::ClassicalValue::Bool(false)
                        )
                    ),
                    None => false,
                }
            }
            StmtKind::OutputDecl { name, .. } => {
                self.bind(name, Binding::Bits);
                false
            }
            StmtKind::InputDecl {
                ty: "bit", name, ..
            } => {
                self.bind(name, Binding::Bits);
                self.local_bits.insert(name);
                true
            }
            StmtKind::InputDecl { name, .. } => {
                let decl = key(stmt);
                self.bind(name, Binding::Var { decl, depth });
                self.runtime.contains(&decl)
            }
            StmtKind::ClassicalDecl {
                constant,
                name,
                value,
                ..
            } => {
                if *constant {
                    self.bind(name, Binding::Constant);
                    return false;
                }
                let decl = key(stmt);
                if value
                    .as_ref()
                    .is_some_and(|value| self.expr_reads(value).any())
                {
                    self.mark(decl);
                }
                self.bind(name, Binding::Var { decl, depth });
                self.runtime.contains(&decl)
            }
            StmtKind::Assign { target, value, .. } => match self.lookup(target) {
                Binding::Bits => {
                    self.needs_call = true;
                    true
                }
                binding => {
                    let reads = self.expr_reads(value);
                    self.write(binding, depth, reads)
                }
            },
            StmtKind::Measure(measure) => {
                if depth == 0
                    && self.conditional == 0
                    && let Some(name) = measure.target.register()
                    && self.local_bits.contains(name)
                    && (measure.target.index.is_none() || self.single_bits.contains(name))
                {
                    self.initialized_bits.insert(name);
                }
                false
            }
            StmtKind::CallAssign(assign) if assign.target.index.is_none() => {
                let Some(target) = assign.target.register() else {
                    return false;
                };
                let dynamic_call = self.functions.contains(assign.name);
                self.needs_call |= dynamic_call;
                let reads = assign.args.iter().fold(
                    Reads {
                        runtime: dynamic_call,
                        bits: false,
                    },
                    |reads, arg| reads.or(self.argument_reads(arg)),
                );
                self.write(self.lookup(target), depth, reads) || dynamic_call
            }
            StmtKind::ArrayDecl(decl) => {
                self.bind(
                    decl.name,
                    Binding::Array {
                        decl: key(stmt),
                        depth,
                    },
                );
                false
            }
            StmtKind::ElementAssign(assign)
                if matches!(self.lookup(assign.array), Binding::Bits) =>
            {
                self.needs_call = true;
                true
            }
            StmtKind::ElementAssign(assign) => {
                let reads = assign
                    .indices
                    .iter()
                    .fold(self.expr_reads(&assign.value), |reads, index| {
                        reads.or(self.expr_reads(index))
                    });
                self.write(self.lookup(assign.array), depth, reads);
                false
            }
            StmtKind::Alias { name, sources } => {
                let bits = sources.iter().any(|source| {
                    source
                        .register()
                        .is_some_and(|name| matches!(self.lookup(name), Binding::Bits))
                });
                self.bind(name, if bits { Binding::Bits } else { Binding::Other });
                false
            }
            StmtKind::Call { name, params, .. } => {
                let dynamic_call = self.functions.contains(*name);
                self.needs_call |= dynamic_call;
                dynamic_call || params.iter().any(|param| self.argument_reads(param).any())
            }
            StmtKind::Expr(value) => {
                let runtime = self.expr_reads(value).runtime;
                self.needs_call |= runtime;
                runtime
            }
            StmtKind::Return(value) => {
                let reads = match value {
                    Some(ast::ReturnValue::Value(value)) => self.expr_reads(value),
                    _ => Reads::default(),
                };
                self.needs_call |= depth > 0
                    || self.scalar_return && reads.any()
                    || !self.scalar_return
                        && matches!(value, Some(ast::ReturnValue::Value(value))
                        if value.as_ident().is_none_or(|name| !self.local_bits.contains(name)));
                true
            }
            StmtKind::If(conditional) => {
                self.conditional += 1;
                let decided = self.decided(&conditional.condition);
                let inner = depth + usize::from(decided != Decided::Static);
                let mut needs = self.scoped(&conditional.then_body, inner);
                if let Some(body) = &conditional.else_body {
                    needs |= self.scoped(body, inner);
                }
                needs |= decided == Decided::Runtime;
                if needs {
                    self.branches.insert(key(stmt));
                }
                self.conditional -= 1;
                needs
            }
            StmtKind::Switch { target, arms } => {
                self.conditional += 1;
                let reads = self.expr_reads(target);
                let mut needs = reads.runtime;
                for arm in arms {
                    needs |= self.scoped(&arm.body, depth + usize::from(reads.any()));
                }
                if needs {
                    self.branches.insert(key(stmt));
                }
                self.conditional -= 1;
                needs
            }
            StmtKind::While { body, .. } => {
                self.needs_call = true;
                self.loops.push(depth + 1);
                let outer_exit = std::mem::replace(&mut self.runtime_exit, false);
                self.scoped(body, depth + 1);
                self.runtime_exit = outer_exit;
                self.loops.pop();
                true
            }
            StmtKind::Break | StmtKind::Continue => {
                let runtime = self.loops.last().is_some_and(|&home| depth > home);
                self.runtime_exit |= runtime;
                runtime
            }
            StmtKind::End => {
                // A def that may end the program runs as a call, so its `end`
                // halts the shot wherever a measurement decides the call.
                self.needs_call = true;
                depth > 0
            }
            StmtKind::For { variable, body, .. } => {
                self.conditional += 1;
                // A pass a measurement may cut short leaves every write in the
                // body, and every pass after it, decided at runtime. A loop
                // found to hold such an exit is walked one level down from the
                // next pass of the fixed point on.
                let unrolled = self.unrolled.contains(&key(stmt));
                let inner = depth + usize::from(unrolled);
                self.loops.push(inner);
                let outer_exit = std::mem::replace(&mut self.runtime_exit, false);
                self.scopes.push(HashMap::new());
                self.bind(variable, Binding::Constant);
                let mut needs = self.block(body, inner);
                self.scopes.pop();
                if self.runtime_exit {
                    if !unrolled {
                        self.unrolled.insert(key(stmt));
                        self.changed = true;
                    }
                    self.needs_call = true;
                    needs = true;
                }
                self.runtime_exit = outer_exit;
                self.loops.pop();
                self.conditional -= 1;
                needs
            }
            StmtKind::Box { body, .. } => self.scoped(body, depth),
            _ => false,
        }
    }
}
